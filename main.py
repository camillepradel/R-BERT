import argparse
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import Trainer, TrainingArguments, set_seed, EvalPrediction, AutoConfig, default_data_collator
from transformers.trainer_utils import is_main_process
from datasets import load_dataset

from utils import load_tokenizer, write_prediction
from model import RBERT
from official_eval import official_f1

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    add_sep_token: bool = field(
        default=False,
        metadata={
            "help": "Add [SEP] token at the end of the sentence"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    data_dir: Optional[str] = field(
        default="./data",
        metadata={"help": "he input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    train_file: Optional[str] = field(
        default="train.txt",
        metadata={"help": "Train file"}
    )
    test_file: Optional[str] = field(
        default="test.txt",
        metadata={"help": "Test file"}
    )
    label_file: Optional[str] = field(
        default="label.txt",
        metadata={"help": "Label file"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout for fully-connected layers"}
        )
    first_layer_to_use: int = field(
        default=6,
        metadata={
            "help": "The index of the lowest layer whose attention heads will be used for classification"
        },
    )
    last_layer_to_use: int = field(
        default=11,
        metadata={
            "help": "The index of the highest layer whose attention heads will be used for classification"
        },
    )
    # TODO: implement behaviour for use_residual_layer
    # use_residual_layer: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Add a skip connection from attention weights to final fc classifier"
    #     },
    # )


@dataclass
class RBertTrainingArguments(TrainingArguments):
    num_train_epochs_frozen: float = field(
        default=0.0,
        metadata={"help": "Number of training epochs with frozen transformer."}
    )
    eval_dir: str = field(
        default="./eval",
        metadata={"help": "Evaluation script, result directory"}
    )


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    
    data_training_args = DataTrainingArguments(
        max_seq_length=args.max_seq_len,
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        label_file=args.label_file,
    )
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        dropout_rate=args.dropout_rate,
        first_layer_to_use=args.first_layer_to_use,
        last_layer_to_use=args.last_layer_to_use,
    )
    training_args = RBertTrainingArguments(
        output_dir=args.model_dir,
        eval_dir=args.eval_dir,
        num_train_epochs=args.num_train_epochs,
        num_train_epochs_frozen=args.num_train_epochs_frozen,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        do_train=args.do_train,
        do_eval=args.do_eval,
        no_cuda=args.no_cuda,
        disable_tqdm=args.do_not_use_tqdm,
        overwrite_output_dir=True,
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # datasets
    datasets = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_training_args.data_dir, data_training_args.train_file),
                "validation": os.path.join(data_training_args.data_dir, data_training_args.test_file),
            },
            delimiter='\t',
            column_names=['label', 'sentence'],
        )

    # Labels
    label_list = [
        label.strip() for label in open(os.path.join(data_training_args.data_dir, data_training_args.label_file), "r", encoding="utf-8")]
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
    )
    tokenizer = load_tokenizer(model_args.model_name_or_path, training_args.output_dir)
    model = RBERT.from_pretrained(model_args.model_name_or_path, config=config, args=model_args)

    label_to_id = {v: i for i, v in enumerate(label_list)}

    # Preprocessing the datasets

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples['sentence'], padding="max_length", max_length=data_training_args.max_seq_length, truncation=True)

        e1_masks = []
        e2_masks = []
        e1_start_id = tokenizer.convert_tokens_to_ids("<e1>")
        e1_end_id = tokenizer.convert_tokens_to_ids("</e1>")
        e2_start_id = tokenizer.convert_tokens_to_ids("<e2>")
        e2_end_id = tokenizer.convert_tokens_to_ids("</e2>")

        for example_input_ids in result['input_ids']:

            e11_p = example_input_ids.index(e1_start_id)  # the start position of entity1
            e12_p = example_input_ids.index(e1_end_id)  # the end position of entity1
            e21_p = example_input_ids.index(e2_start_id)  # the start position of entity2
            e22_p = example_input_ids.index(e2_end_id)  # the end position of entity2

            # TODO: take into account add_sep_token param

            # e1 mask, e2 mask
            e1_mask = [0] * data_training_args.max_seq_length
            e2_mask = [0] * data_training_args.max_seq_length

            try:
                for i in range(e11_p, e12_p + 1):
                    e1_mask[i] = 1
            except IndexError:
                # entities can appear in the sentence after data_training_args.max_seq_length
                # in that case, we can't set any mask
                logger.debug("Entity 1 appears beyond max_seq_length.")

            try:
                for i in range(e21_p, e22_p + 1):
                    e2_mask[i] = 1
            except IndexError:
                # entities can appear in the sentence after data_training_args.max_seq_length
                # in that case, we can't set any mask
                logger.debug("Entity 2 appears beyond max_seq_length.")
            
            e1_masks.append(e1_mask)
            e2_masks.append(e2_mask)


        result["e1_mask"] = e1_masks
        result["e2_mask"] = e2_masks
        result["label"] = [label_to_id[l] for l in examples["label"]]

        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_training_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # custom compute_metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        write_prediction(label_list, os.path.join(training_args.eval_dir, "proposed_answers.txt"), preds)
        return {
            "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
            "f1": official_f1(),
        }

    # TODO:
    #  - implement behaviour with num_train_epochs_frozen
    #  - setup original scheduler and weight decay behaviour
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info(f"***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="Evaluation script, result directory",
    )
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model Name or Path",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--first_layer_to_use",
        default=6,
        type=int,
        help="The index of the lowest layer whose attention heads will be used for classification",
    )
    parser.add_argument(
        "--last_layer_to_use",
        default=11,
        type=int,
        help="The index of the highest layer whose attention heads will be used for classification",
    )
    # parser.add_argument(
    #     "--use_residual_layer",
    #     action="store_true",
    #     help="Add a skip connection from attention weights to final fc classifier",
    # )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument( # NOT DONE
        "--num_train_epochs_frozen",
        default=0.0,
        type=float,
        help="Number of training epochs with frozen transformer.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )
    parser.add_argument(
        "--do_not_use_tqdm",
        action="store_true",
        help="Whether to disable fancy progress bars",
    )

    args = parser.parse_args()

    main(args)
