import os
import fileinput

from transformers import AutoTokenizer, AutoConfig

from official_eval import official_f1

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def load_tokenizer(model_name_or_path, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tokenizer.save_pretrained(model_dir)
    config.save_pretrained(model_dir)
    with fileinput.FileInput(os.path.join(model_dir, 'vocab.txt'), inplace=True) as file:
        for line in file:
            line = line.replace('[unused1]', ADDITIONAL_SPECIAL_TOKENS[0])
            line = line.replace('[unused2]', ADDITIONAL_SPECIAL_TOKENS[1])
            line = line.replace('[unused3]', ADDITIONAL_SPECIAL_TOKENS[2])
            line = line.replace('[unused4]', ADDITIONAL_SPECIAL_TOKENS[3])
            print(line, end='')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(relation_labels, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))
