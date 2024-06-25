import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union
from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel


class DefSentPlus(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        backbone_model_name: str,
        device: torch.device = None,
    ) -> None:
        super(DefSentPlus, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.backbone_model_name = backbone_model_name

        self.__validate_backbone_model(backbone_model_name=self.backbone_model_name)
        self.encoder, self.tokenizer = self.__get_encoder_tokenizer(backbone_model_name=self.backbone_model_name,
                                                                    model_name_or_path=self.model_name_or_path)

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.encoder = self.encoder.to(device)

    def __validate_backbone_model(self, backbone_model_name) -> None:
        available_model_names = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
        if backbone_model_name not in available_model_names:
            raise RuntimeError("Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.")

    def __get_encoder_tokenizer(self, backbone_model_name, model_name_or_path):
        if "roberta" in backbone_model_name:
            tokenizer = RobertaTokenizer.from_pretrained(backbone_model_name)
            model = RobertaModel.from_pretrained(model_name_or_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(backbone_model_name, do_lower_case=True)
            model = BertModel.from_pretrained(model_name_or_path)

        return model, tokenizer

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[str, List[str]],
        pooling: str,
        batch_size: int = 16,
    ) -> Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        if pooling == "prompt":
            if "roberta" in self.backbone_model_name:
                sentences = [sentence.replace(sentence, '''This sentence : "''' + sentence + '''" means <mask> . ''') for sentence in sentences]
            else:
                sentences = [sentence.replace(sentence, '''This sentence : "''' + sentence + '''" means [MASK] . ''') for sentence in sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        data_loader = torch.utils.data.DataLoader(
            list(zip(inputs.input_ids, inputs.attention_mask)),
            batch_size=batch_size,
        )

        all_embs = []
        for input_ids, attention_mask in data_loader:
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            token_representations = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

            embs = self.__get_pooling_results(token_representations, input_ids=input_ids, attention_mask=attention_mask, pooling=pooling)
            # Prevent overuse of memory.
            embs = embs.cpu()
            all_embs.append(embs)

        embeddings = torch.cat(all_embs, dim=0)
        return embeddings

    def __get_pooling_results(self, token_representations, input_ids, attention_mask, pooling):
        if pooling == "cls":
            return token_representations[:, 0]
        elif pooling == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_representations.size()).float()
            return torch.sum(token_representations * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif pooling == "prompt":
            return token_representations[input_ids == self.tokenizer.mask_token_id]
        else:
            raise ValueError("Available pooling name is cls, mean, or prompt.")