import logging
import os
import sys
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    AutoConfig,
    default_data_collator,
    HfArgumentParser,
    TrainerCallback,
)
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    data_dir: Optional[str] = field(
        default="./data",
        metadata={"help": "he input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    train_file: Optional[str] = field(
        default="train.tsv",
        metadata={"help": "Train file"}
    )
    test_file: Optional[str] = field(
        default="test.tsv",
        metadata={"help": "Test file"}
    )
    label_file: Optional[str] = field(
        default="label.txt",
        metadata={"help": "Label file"}
    )

# TODO: should be logged and used for hyperparameter tunning in wandb
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
    fc1_d1_layer_output_size: int = field(
        default=100,
        metadata={
            "help": "The size of the output of fc layer applied to depth-1 attentions. "
            "Set to zero to not use this fc layer"
        },
    )
    fc1_d2_layer_output_size: int = field(
        default=500,
        metadata={
            "help": "The size of the output of fc layer applied to depth-2 attentions. "
            "Set to zero to not use this fc layer"
        },
    )
    fc2_layer_output_size: int = field(
        default=100,
        metadata={
            "help": "The size of the output of fc layer applied to first fc layers output (or to attentions if the latter are disabled). "
            "Set to zero to not use this fc layer"
        },
    )
    skip_1_d1: bool = field(
        default=False,
        metadata={
            "help": "Add a skip connection (residual layer) from depth-1 attention weights to second fc layer (skip fc1_d1) "
            "Cannot be set to True if fc1_d1 layer is disabled (fc1_d1_layer_output_size==0)"
        },
    )
    skip_1_d2: bool = field(
        default=False,
        metadata={
            "help": "Add a skip connection (residual layer) from max pooled depth-2 attention weights to second fc layer (skip fc1_d2) "
            "Cannot be set to True if fc1_d2 layer is disabled (fc1_d2_layer_output_size==0)"
        },
    )
    skip_2_d1: bool = field(
        default=False,
        metadata={
            "help": "Add a skip connection (residual layer) from depth-1 attention weights to classifier layer (skip fc1_d1 and fc2) "
            "Cannot be set to True if fc2 layer is disabled (fc2_layer_output_size==0)"
        },
    )
    skip_2_d2: bool = field(
        default=False,
        metadata={
            "help": "Add a skip connection (residual layer) from max pooled depth-2 attention weights to classifier layer (skip fc1_d2 and fc2) "
            "Cannot be set to True if fc2 layer is disabled (fc2_layer_output_size==0)"
        },
    )
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
    exclude_markup_tokens_from_masks: bool = field(
        default=False,
        metadata={
            "help": "Exclude the markup tokens (<e1>... </e2>) from the entity masks."
        },
    )
    label_smoothing_epsilon: float = field(
        default=0.0,
        metadata={
            "help": "Epsilon used for label smoothing in loss function. "
            "Set to 0 to disable label smoothing."
        },
    )


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
    # override default values from TrainingArguments
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for Adam."})
    num_train_epochs: float = field(default=10.0, metadata={"help": "Total number of training epochs to perform."})


# allows to freeze transformer weights during num_train_epochs_frozen epochs before unfreezing them
class FreezeTransformerCallback(TrainerCallback):

    def __init__(self, training_args):
        self.num_train_epochs_frozen = training_args.num_train_epochs_frozen
        self.transformer_frozen = False

    def _set_transformer_trainable(self, model, requires_grad=True):
        for param in model.bert.parameters():
            param.requires_grad = requires_grad

    def on_epoch_begin(self, args, state, control, logs=None, **kwargs):
        if state.epoch < self.num_train_epochs_frozen and not self.transformer_frozen:
            # freeze transformer weights
            self._set_transformer_trainable(kwargs['model'], False)
            self.transformer_frozen = True
        elif state.epoch >= self.num_train_epochs_frozen and self.transformer_frozen:
            # unfreeze transformer weights
            self._set_transformer_trainable(kwargs['model'], True)
            self.transformer_frozen = False


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    # adapt parameters generated by wandb sweeps to HfArgumentParser
    i_arg = len(sys.argv)-1
    while i_arg >= 0:
        if sys.argv[i_arg].endswith('=true'):
            sys.argv[i_arg] = sys.argv[i_arg].replace('=true', '')
        elif sys.argv[i_arg].endswith('=false'):
            del sys.argv[i_arg]
        i_arg -= 1
    print('sys.argv', sys.argv)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, RBertTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_training_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_training_args, training_args = parser.parse_args_into_dataclasses()
    
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
        result = tokenizer(examples['sentence'], padding="max_length", max_length=model_args.max_seq_length, truncation=True)

        e1_masks = []
        e2_masks = []
        e1_start_id = tokenizer.convert_tokens_to_ids("<e1>")
        e1_end_id = tokenizer.convert_tokens_to_ids("</e1>")
        e2_start_id = tokenizer.convert_tokens_to_ids("<e2>")
        e2_end_id = tokenizer.convert_tokens_to_ids("</e2>")

        for i, example_input_ids in enumerate(result['input_ids']):

            # take into account add_sep_token param
            if not model_args.add_sep_token:
                i_sep_token = example_input_ids.index(tokenizer.sep_token_id)
                result['input_ids'][i][i_sep_token] = tokenizer.pad_token_id
                result['attention_mask'][i][i_sep_token] = 0

            e11_p = example_input_ids.index(e1_start_id)  # the start position of entity1
            e12_p = example_input_ids.index(e1_end_id)  # the end position of entity1
            e21_p = example_input_ids.index(e2_start_id)  # the start position of entity2
            e22_p = example_input_ids.index(e2_end_id)  # the end position of entity2

            if model_args.exclude_markup_tokens_from_masks:
                e11_p += 1
                e12_p -= 1
                e21_p += 1
                e22_p -= 1

            # e1 mask, e2 mask
            e1_mask = [0] * model_args.max_seq_length
            e2_mask = [0] * model_args.max_seq_length

            try:
                for i in range(e11_p, e12_p + 1):
                    e1_mask[i] = 1
            except IndexError:
                # entities can appear in the sentence after model_args.max_seq_length
                # in that case, we can't set any mask
                logger.debug("Entity 1 appears beyond max_seq_length.")

            try:
                for i in range(e21_p, e22_p + 1):
                    e2_mask[i] = 1
            except IndexError:
                # entities can appear in the sentence after model_args.max_seq_length
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[FreezeTransformerCallback(training_args)]
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
    main()
