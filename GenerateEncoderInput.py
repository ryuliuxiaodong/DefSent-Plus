import torch
from transformers import BertTokenizer, RobertaTokenizer

import sys
from collections import OrderedDict


def load_sys_args():
    argv = sys.argv[1:]
    sys_args = OrderedDict()

    if len(argv) > 0:
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            sys_args[arg_name] = arg_value
    else:
        raise RuntimeError('No input error')

    return sys_args

def ensure_hyperparameters_type(sys_args):
    hyperparameters = OrderedDict()

    try:
        hyperparameters["backbone_model_name"] = str(sys_args["backbone_model_name"])
    except:
        raise RuntimeError('backbone model name is missing.')
    try:
        hyperparameters["max_seq_length"] = int(sys_args["max_seq_length"])
    except:
        hyperparameters["max_seq_length"] = 150
    try:
        hyperparameters["dataset_file"] = str(sys_args["dataset_file"])
    except:
        raise RuntimeError('dataset file is missing.')
    try:
        hyperparameters["dataset_file_encoding"] = str(sys_args["dataset_file_encoding"])
    except:
        raise RuntimeError('dataset file encoding (urf-8, urf-16, etc.) is missing.')
    try:
        hyperparameters["save_path"] = str(sys_args["save_path"])
    except:
        raise RuntimeError('save path is missing.')

    return hyperparameters


def get_and_save_inputs(definitions_from_all_entries, tokenizer, max_seq_length, save_path, backbone_model_name):
    entry_ids = []
    count = 0

    input_ids = []
    token_type_ids = []
    attention_mask = []

    for descriptions_of_this_entry in definitions_from_all_entries:
        for description in descriptions_of_this_entry:
            entry_ids.append(count)

            encodings = tokenizer.encode_plus(description, padding='max_length', max_length=max_seq_length)
            input_ids.append(encodings['input_ids'])
            attention_mask.append(encodings['attention_mask'])
            if "roberta" not in backbone_model_name:
                token_type_ids.append(encodings['token_type_ids'])

        print(count)
        count += 1

    if "roberta" not in backbone_model_name:
        assert len(entry_ids) == len(input_ids) == len(token_type_ids) == len(attention_mask), "inconsistent"
    else:
        assert len(entry_ids) == len(input_ids) == len(attention_mask), "inconsistent"
    print("total processed sentences: " + str(len(entry_ids)))

    entry_ids = torch.LongTensor([entry_ids]).T
    torch.save(entry_ids, save_path + "/entry_id_labels.pth")
    input_ids = torch.LongTensor(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)
    torch.save(input_ids, save_path + "/input_ids.pth")
    torch.save(attention_mask, save_path + "/attention_mask.pth")
    if "roberta" not in backbone_model_name:
        torch.save(token_type_ids, save_path + "/token_type_ids.pth")



def get_definitions(definition):
    definitions = definition.split('<|list_more_def|>')
    return definitions

def preprocess_dataset(dataset_file, dataset_file_encoding):
    definitions_from_all_entries = []

    # D:\Dataset\Dictionary\Ox+WN.csv
    word2definitions_file = open(dataset_file, 'r', encoding=dataset_file_encoding)
    word2definitions_list = word2definitions_file.readlines()
    word2definitions_file.close()

    for record in word2definitions_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')

        definitions = get_definitions(all_values[1])
        definitions_from_all_entries.append(definitions)

    return definitions_from_all_entries

if __name__ == "__main__":
    sys_args = load_sys_args()
    hyperparameters = ensure_hyperparameters_type(sys_args=sys_args)

    available_model_names = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
    if hyperparameters["backbone_model_name"] not in available_model_names:
        raise RuntimeError("Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.")

    if "roberta" in hyperparameters["backbone_model_name"]:
        tokenizer = RobertaTokenizer.from_pretrained(hyperparameters["backbone_model_name"])
    else:
        tokenizer = BertTokenizer.from_pretrained(hyperparameters["backbone_model_name"], do_lower_case=True)

    definitions_from_all_entries = preprocess_dataset(hyperparameters["dataset_file"],
                                                      hyperparameters["dataset_file_encoding"])

    get_and_save_inputs(definitions_from_all_entries=definitions_from_all_entries,
                        tokenizer=tokenizer,
                        max_seq_length=hyperparameters["max_seq_length"],
                        save_path=hyperparameters["save_path"],
                        backbone_model_name=hyperparameters["backbone_model_name"])

