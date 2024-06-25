import torch
from torch import nn
from transformers import BertModel, RobertaModel, RobertaForMaskedLM

import sys
from collections import OrderedDict
import time
import datetime

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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
        hyperparameters["sentence_embed_encoding"] = str(sys_args["sentence_embed_encoding"])
        if hyperparameters["sentence_embed_encoding"] != "cls" and hyperparameters["sentence_embed_encoding"] != "mean":
            raise RuntimeError('the encoding type of sentence embeddings should be cls or mean.')
    except:
        raise RuntimeError('the encoding type of sentence embeddings is missing.')
    try:
        hyperparameters["input_file_path"] = str(sys_args["input_file_path"])
    except:
        raise RuntimeError('the path to input file is missing.')
    try:
        hyperparameters["entry_embed"] = str(sys_args["entry_embed"])
    except:
        raise RuntimeError('the weight matrix to entry embeddings is missing.')
    try:
        hyperparameters["save_path"] = str(sys_args["save_path"])
    except:
        raise RuntimeError('save path is missing.')
    try:
        hyperparameters["batch_size"] = int(sys_args["batch_size"])
    except:
        raise RuntimeError('batch size is missing.')
    try:
        hyperparameters["learning_rate"] = float(sys_args["learning_rate"])
    except:
        raise RuntimeError('learning rate is missing.')

    return hyperparameters


def train_and_save_encoder(weight_matrix, input_file_path, backbone_model_name, model, save_path, encoding, batch_size, learning_rate):
    embedding_weight_data = torch.load(weight_matrix)
    weight_size = list(embedding_weight_data.size())
    embed_nums = weight_size[0]
    dims = weight_size[1]
    print(embed_nums, dims)

    embeddings = nn.Embedding(embed_nums, dims)

    embeddings.weight.data = embedding_weight_data
    embeddings.weight.requires_grad = False
    embeddings.cuda()

    prediction_head = nn.Linear(dims, embed_nums, bias=False)
    prediction_head.weight = embeddings.weight
    prediction_head.cuda()

    batch_size = batch_size
    epochs = 1

    input_ids = torch.load(input_file_path + "/input_ids.pth")
    attention_mask = torch.load(input_file_path + "/attention_mask.pth")
    target_entry_ids = torch.load(input_file_path + "/entry_id_labels.pth")
    if "roberta" not in backbone_model_name:
        token_type_ids = torch.load(input_file_path + "/token_type_ids.pth")
        train_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, target_entry_ids)
    else:
        train_dataset = TensorDataset(input_ids, attention_mask, target_entry_ids)

    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    ### copy weight of dense layer in MLM to pooler
    if "roberta" in backbone_model_name:
        lm = RobertaForMaskedLM.from_pretrained(backbone_model_name)
        model.pooler.dense.weight.data.copy_(lm.lm_head.dense.weight.data)
        model.pooler.dense.bias.data.copy_(lm.lm_head.dense.bias.data)
        model.pooler.activation = nn.GELU()
        model.cuda()
        del lm
    model.cuda()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=learning_rate,
                      eps=1e-6,
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    loss_func = nn.CrossEntropyLoss()

    avg_epoch_loss = []

    if "roberta" not in backbone_model_name:

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            model.train()

            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].cuda()
                b_token_type_ids = batch[1].cuda()
                b_attention_mask = batch[2].cuda()

                b_target_entry_ids = batch[3].cuda()

                model.zero_grad()
                optimizer.zero_grad()

                if encoding == "cls":
                    sentence_representations = model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)[1]
                else:
                    representations = model(input_ids=b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_mask)[0]
                    input_mask_expanded = b_attention_mask.unsqueeze(-1).expand(representations.size()).float()
                    description_representation = torch.sum(representations * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    sentence_representations = model.pooler(description_representation.unsqueeze(1))

                prediction_scores = prediction_head(sentence_representations)
                loss = loss_func(prediction_scores, b_target_entry_ids.view(-1))

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)

                optimizer.step()

                scheduler.step()

                if step % 400 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    avg_train_loss = total_train_loss / step
                    print("  400-step training loss: {0:.5f}".format(avg_train_loss))

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_epoch_loss.append(avg_train_loss)

            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.5f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

        model.eval()
        model.save_pretrained(save_path)

    else:

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            model.train()

            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].cuda()
                b_attention_mask = batch[1].cuda()

                b_target_entry_ids = batch[2].cuda()

                model.zero_grad()
                optimizer.zero_grad()

                if encoding == "cls":
                    sentence_representations = model(input_ids=b_input_ids, attention_mask=b_attention_mask)[1]
                else:
                    representations = model(input_ids=b_input_ids, attention_mask=b_attention_mask)[0]
                    input_mask_expanded = b_attention_mask.unsqueeze(-1).expand(representations.size()).float()
                    description_representation = torch.sum(representations * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    sentence_representations = model.pooler(description_representation.unsqueeze(1))

                prediction_scores = prediction_head(sentence_representations)
                loss = loss_func(prediction_scores, b_target_entry_ids.view(-1))

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)

                optimizer.step()

                scheduler.step()

                if step % 400 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    avg_train_loss = total_train_loss / step
                    print("  400-step training loss: {0:.5f}".format(avg_train_loss))

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_epoch_loss.append(avg_train_loss)

            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.5f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

        model.eval()
        model.save_pretrained(save_path)


if __name__ == "__main__":
    sys_args = load_sys_args()
    hyperparameters = ensure_hyperparameters_type(sys_args=sys_args)

    available_model_names = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
    if hyperparameters["backbone_model_name"] not in available_model_names:
        raise RuntimeError("Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.")

    if "roberta" in hyperparameters["backbone_model_name"]:
        model = RobertaModel.from_pretrained(hyperparameters["model_path"])
    else:
        model = BertModel.from_pretrained(hyperparameters["model_path"])
    model.eval()
    model.cuda()

    train_and_save_encoder(weight_matrix=hyperparameters["entry_embed"],
                           input_file_path=hyperparameters["input_file_path"],
                           backbone_model_name=hyperparameters["backbone_model_name"],
                           model=model,
                           save_path=hyperparameters["save_path"],
                           encoding=hyperparameters["sentence_embed_encoding"],
                           batch_size=hyperparameters["batch_size"],
                           learning_rate=hyperparameters["learning_rate"])

