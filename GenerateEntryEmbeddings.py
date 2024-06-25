import torch
from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel

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
        hyperparameters["model_path"] = str(sys_args["model_path"])
    except:
        raise RuntimeError('model path is missing.')
    try:
        hyperparameters["entry_embed_encoding"] = str(sys_args["entry_embed_encoding"])
        if hyperparameters["entry_embed_encoding"] != "ac" and hyperparameters["entry_embed_encoding"] != "amp":
            raise RuntimeError('the encoding type of entry embeddings should be ac or amp.')
    except:
        raise RuntimeError('the encoding type of entry embeddings is missing.')
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


def get_and_save_entry_embedding_weight(definitions_from_all_entries, model, tokenizer, definitions_max_len, save_path, backbone_model_name, entry_embed_encoding):
    all_average = []

    count = 0
    for definitions in definitions_from_all_entries:
        print(count)
        ### if an entry has multiple definitions
        if len(definitions) > 1:
            outputs = tokenizer(definitions, max_length=definitions_max_len[count], truncation=True, padding='max_length')
            input_ids = torch.LongTensor(outputs["input_ids"]).cuda()
            attention_mask = torch.LongTensor(outputs["attention_mask"]).cuda()
            if "roberta" not in backbone_model_name:
                token_type_ids = torch.LongTensor(outputs["token_type_ids"]).cuda()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if entry_embed_encoding == "ac":
                average_cls_of_this_definitions = torch.mean(outputs[0][:, 0], dim=0)
                all_average.append(average_cls_of_this_definitions.cpu().tolist())
            else:
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                description_representation = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                description_representation = torch.mean(description_representation, dim=0)
                all_average.append(description_representation.cpu().tolist())
        ### if an entry has only one definition
        else:
            token_ids = tokenizer.encode(definitions[0])
            input_ids = [token_ids]
            input_ids = torch.LongTensor(input_ids).cuda()
            token_type_ids = [[0 for k in range(len(token_ids))]]
            attention_mask = [[1 for k in range(len(token_ids))]]
            attention_mask = torch.LongTensor(attention_mask).cuda()
            if "roberta" not in backbone_model_name:
                token_type_ids = torch.LongTensor(token_type_ids).cuda()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if entry_embed_encoding == "ac":
                average_cls_of_this_definitions = outputs[0][0][0]
                all_average.append(average_cls_of_this_definitions.cpu().tolist())
            else:
                description_representation = torch.mean(outputs[0][0], dim=0)
                all_average.append(description_representation.cpu().tolist())

        count += 1

    print("total entry embeddings: " + str(count))

    all_average = torch.tensor(all_average, dtype=torch.float)
    if entry_embed_encoding == "ac":
        torch.save(all_average, save_path + "/AC_entry_embed_weight.pth")
    else:
        torch.save(all_average, save_path + "/AMP_entry_embed_weight.pth")

def get_definitions(definition):
    definitions = definition.split('<|list_more_def|>')
    return definitions

def get_max_len_for_definitions(definitions, tokenizer):
    tmp_max_len = 0
    for definition in definitions:
        tmp_len = len(tokenizer.tokenize(definition))
        if tmp_len > tmp_max_len:
            tmp_max_len = tmp_len

    tmp_max_len += 2
    return tmp_max_len

def preprocess_dataset(dataset_file, dataset_file_encoding):
    definitions_from_all_entries = []
    definitions_max_len = []

    # D:\Dataset\Dictionary\Ox+WN.csv
    word2definitions_file = open(dataset_file, 'r', encoding=dataset_file_encoding)
    word2definitions_list = word2definitions_file.readlines()
    word2definitions_file.close()

    for record in word2definitions_list:
        record = record.rstrip("\n")
        all_values = record.split('\t')

        definitions = get_definitions(all_values[1])
        definitions_from_all_entries.append(definitions)
        definitions_max_len.append(get_max_len_for_definitions(definitions, tokenizer=tokenizer))

    return definitions_from_all_entries, definitions_max_len

if __name__ == "__main__":
    sys_args = load_sys_args()
    hyperparameters = ensure_hyperparameters_type(sys_args=sys_args)

    available_model_names = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
    if hyperparameters["backbone_model_name"] not in available_model_names:
        raise RuntimeError("Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.")

    if "roberta" in hyperparameters["backbone_model_name"]:
        tokenizer = RobertaTokenizer.from_pretrained(hyperparameters["backbone_model_name"])
        model = RobertaModel.from_pretrained(hyperparameters["model_path"])
    else:
        tokenizer = BertTokenizer.from_pretrained(hyperparameters["backbone_model_name"], do_lower_case=True)
        model = BertModel.from_pretrained(hyperparameters["model_path"])
    model.eval()
    model.cuda()

    definitions_from_all_entries, definitions_max_len = preprocess_dataset(hyperparameters["dataset_file"],
                                                                           hyperparameters["dataset_file_encoding"])

    get_and_save_entry_embedding_weight(definitions_from_all_entries=definitions_from_all_entries,
                                        model=model,
                                        tokenizer=tokenizer,
                                        definitions_max_len=definitions_max_len,
                                        save_path=hyperparameters["save_path"],
                                        backbone_model_name=hyperparameters["backbone_model_name"],
                                        entry_embed_encoding=hyperparameters["entry_embed_encoding"])

