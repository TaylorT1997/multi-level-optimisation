import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import json
import torch

import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    AutoModel,
    AutoModelForTokenClassification,
    BertForTokenClassification,
    PreTrainedModel,
    RobertaModel,
    BertModel,
    BertPreTrainedModel,
)

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import DataCollatorForLanguageModeling

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from utils.tsv_dataset import (
    TSVClassificationDataset,
    Split,
    get_labels,
    compute_seq_classification_metrics,
    MaskedDataCollator,
)

# from utils.arguments import datasets, DataTrainingArguments, ModelArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftAttentionSeqClassModel(nn.Module):
    def __init__(
        self,
        config_dict,
        bert_out_size,
        num_labels,
        token_supervision=True,
        sequence_supervision=True,
        use_only_token_attention=False,
        use_multi_head_attention=False,
        supervised_heads=[1, 2, 3],
        debug=False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        set_seed(config_dict["seed"])

        print(config_dict)

        self.initializer_name = config_dict["initializer_name"]
        self.square_attention = config_dict["square_attention"]
        self.num_labels = num_labels
        self.token_supervision = config_dict["token_supervision"]
        self.sequence_supervision = config_dict["sequence_supervision"]
        self.use_only_token_attention = config_dict["use_only_token_attention"]
        self.use_multi_head_attention = config_dict["use_multi_head_attention"]
        self.supervised_heads = config_dict["supervised_heads"]

        self.alpha = config_dict["soft_attention_alpha"]
        self.gamma = config_dict["soft_attention_gamma"]
        self.beta = config_dict["soft_attention_beta"]

        self.dropout = nn.Dropout(p=config_dict["hid_to_attn_dropout"])

        self.attention_evidence = nn.Linear(
            bert_out_size, config_dict["attention_evidence_size"]
        )  # layer for predicting attention weights
        self.attention_weights = nn.Linear(config_dict["attention_evidence_size"], 1)

        self.final_hidden = nn.Linear(
            bert_out_size, config_dict["final_hidden_layer_size"],
        )
        self.result_layer = nn.Linear(
            config_dict["final_hidden_layer_size"], self.num_labels
        )

        self.model_name = config_dict["model_name"]

        if config_dict["attention_activation"] == "sharp":
            self.attention_act = torch.exp
        elif config_dict["attention_activation"] == "soft":
            self.attention_act = torch.sigmoid
        elif config_dict["attention_activation"] == "linear":
            pass
        else:
            raise ValueError(
                "Unknown activation for attention: "
                + str(self.config["attention_activation"])
            )
        self.apply(self.init_weights)

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def init_weights(self, m):
        if self.initializer_name == "normal":
            self.initializer = nn.init.normal_
        elif self.initializer_name == "glorot":
            self.initializer = nn.init.xavier_normal_
        elif self.initializer_name == "xavier":
            self.initializer = nn.init.xavier_uniform_

        if isinstance(m, nn.Linear):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        bert_outputs,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        token_labels=None,
        offset_mapping=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_scores=None,
        **kwargs,
    ):
        inp_lengths = (attention_mask != 0).sum(dim=1) - 1
        bert_length = input_ids.shape[1] - 1
        bert_hidden_outputs = bert_outputs.last_hidden_state[:, 1:]
        after_dropout = self.dropout(bert_hidden_outputs)

        if self.use_multi_head_attention:
            last_pretrained_layer = bert_outputs.attentions[-1]
            cls_attentions = last_pretrained_layer[:, :, 0, :]

            aggregated_head_attentions = None
            for head in self.supervised_heads:
                head_attention = last_pretrained_layer[:, head, 0, :]
                if aggregated_head_attentions == None:
                    aggregated_head_attentions = head_attention
                else:
                    aggregated_head_attentions += head_attention

            aggregated_head_attentions /= len(self.supervised_heads)

            attn_weights = aggregated_head_attentions[:, 1:]
            attn_weights = torch.where(
                self._sequence_mask(inp_lengths, maxlen=bert_length),
                attn_weights,
                torch.zeros_like(attn_weights),
            )

            if self.debug:
                print(f"last_pretrained_layer shape: \n{last_pretrained_layer.shape}\n")
                print(f"last_pretrained_layer: \n{last_pretrained_layer}\n")

                print(f"cls_attentions shape: \n{cls_attentions.shape}\n")
                print(f"cls_attentions: \n{cls_attentions}\n")

                print(
                    f"aggregated_head_attentions shape: \n{aggregated_head_attentions.shape}\n"
                )
                print(f"aggregated_head_attentions: \n{aggregated_head_attentions}\n")
        else:
            attn_evidence = torch.tanh(self.attention_evidence(after_dropout))
            attn_weights = self.attention_weights(attn_evidence)

            if self.debug:
                print(f"bert_hidden_outputs shape: \n{bert_hidden_outputs.shape}\n")
                print(f"bert_hidden_outputs: \n{bert_hidden_outputs}\n")

                print(f"attn_evidence shape: \n{attn_evidence.shape}\n")
                print(f"attn_evidence: \n{attn_evidence}\n")

                print(f"attn_weights shape: \n{attn_weights.shape}\n")
                print(f"attn_weights: \n{attn_weights}\n")
            attn_weights = attn_weights.view(
                bert_hidden_outputs.size()[:2]
            )  # batch_size, seq_length

            attn_weights = self.attention_act(attn_weights)

            if self.debug:
                print(f"attn_weights_after_activation shape: \n{attn_weights.shape}\n")
                print(f"attn_weights_after_activation: \n{attn_weights}\n")

            attn_weights = torch.where(
                self._sequence_mask(inp_lengths, maxlen=bert_length),
                attn_weights,
                torch.zeros_like(attn_weights),  # seq length
            )

            if self.debug:
                print(f"attn_weights_masked shape: \n{attn_weights.shape}\n")
                print(f"attn_weights_masked: \n{attn_weights}\n")

        self.attention_weights_unnormalised = attn_weights
        if self.square_attention:
            attn_weights = torch.square(attn_weights)
        # normalise attn weights
        attn_weights = attn_weights / torch.sum(attn_weights, dim=1, keepdim=True)
        self.attention_weights_normalised = attn_weights

        if self.debug:
            print(f"attn_weights_normalised shape: \n{attn_weights.shape}\n")
            print(f"attn_weights_normalised: \n{attn_weights}\n")

        if self.use_only_token_attention:
            max_token_attention = torch.max(self.attention_weights_unnormalised, dim=1)
            self.sentence_scores = max_token_attention.values.unsqueeze(1)
        else:
            proc_tensor = torch.bmm(
                after_dropout.transpose(1, 2), attn_weights.unsqueeze(2)
            ).squeeze(dim=2)
            proc_tensor = torch.tanh(self.final_hidden(proc_tensor))

            self.sentence_scores = torch.sigmoid(self.result_layer(proc_tensor))
            self.sentence_scores = self.sentence_scores.view(
                [bert_hidden_outputs.shape[0], self.num_labels]
            )

            if self.debug:
                print(f"proc_tensor shape: \n{proc_tensor.shape}\n")
                print(f"proc_tensor: \n{proc_tensor}\n")

                print(f"self.sentence_scores shape: \n{self.sentence_scores.shape}\n")
                print(f"self.sentence_scores: \n{self.sentence_scores}\n")

        if self.debug:
            print(f"self.sentence_scores shape: \n{self.sentence_scores.shape}\n")
            print(f"self.sentence_scores: \n{self.sentence_scores}\n")

        outputs = (
            self.sentence_scores,
            torch.cat(
                [
                    torch.zeros((input_ids.shape[0], 1)).to(self.device),
                    self.attention_weights_unnormalised,
                ],
                dim=1,
            ),
        )
        # loss = torch.tensor(0, device=self.device)
        if labels is not None:
            # SEQUENCE LOSS
            if self.sequence_supervision:
                # if self.sentence_scores.shape[1] == 1:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(self.sentence_scores.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        self.sentence_scores.view(-1, self.num_labels), labels.view(-1)
                    )
            if self.debug:
                print(f"num_labels: \n{self.num_labels}\n")
                print(f"labels: \n{labels}\n")
                print(f"loss: \n{loss}\n")

            if self.alpha != 0:
                min_attentions, _ = torch.min(
                    torch.where(
                        self._sequence_mask(inp_lengths, maxlen=bert_length),
                        self.attention_weights_unnormalised,
                        torch.zeros_like(self.attention_weights_unnormalised) + 1e6,
                    ),  # [:, 1:],
                    dim=1,
                )
                l2 = self.alpha * torch.mean(torch.square(min_attentions.view(-1)))

                if loss == None:
                    loss = l2
                else:
                    loss += l2

            if self.debug:
                print(f"alpha min_attentions: \n{min_attentions}\n")
                print(f"alpha l2: \n{l2}\n")

            if self.gamma != 0:
                # don't include 0 for CLS token
                attn_weights_masked = torch.where(
                    self._sequence_mask(inp_lengths, maxlen=bert_length),
                    self.attention_weights_unnormalised,
                    torch.zeros_like(self.attention_weights_unnormalised) - 1e6,
                )
                max_attentions, _ = torch.max(attn_weights_masked, dim=1,)
                l3 = self.gamma * torch.mean(
                    torch.square(max_attentions.view(-1) - labels.view(-1))
                )
                loss += l3

            if self.debug:
                print(f"gamma attn_weights_masked: \n{attn_weights_masked}\n")
                print(f"gamma max_attentions: \n{max_attentions}\n")
                print(f"gamma l3: \n{l3}\n")

                print(f"token_labels: \n{token_labels}\n")

            if self.token_supervision == True:
                word_attentions = self._apply_subword_method(
                    self.attention_weights_unnormalised, offset_mapping
                )

                token_labels = token_labels[:, 1:]
                zero_labels = torch.where(
                    token_labels == 1, token_labels, torch.zeros_like(token_labels)
                )
                masked_token_attention = torch.where(
                    ((token_labels == 0) | (token_labels == 1)),
                    self.attention_weights_unnormalised,
                    torch.zeros_like(token_labels),
                )
                mse_loss = nn.MSELoss(reduction="mean")
                token_loss = mse_loss(masked_token_attention, zero_labels)

                loss += token_loss

                if self.debug:
                    print(f"zero_labels: \n{zero_labels}\n")
                    print(f"masked_token_attention: \n{masked_token_attention}\n")
                    print(f"token_loss: \n{token_loss}\n")

            outputs = (loss,) + outputs

            if self.debug:
                print(f"outputs: \n{outputs}\n")
                sys.exit()

        return outputs

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)

        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def _apply_subword_method(
        self, token_attention_output, offset_mapping, subword_method="max"
    ):
        word_attention_output = token_attention_output.clone()

        if "bert-base" in self.model_name:
            individual_subword_indices = (offset_mapping[:, :, 0] != 0).nonzero(
                as_tuple=False
            )
        elif "deberta-base" in self.model_name:
            individual_subword_indices = (offset_mapping[:, :, 0] != 0).nonzero(
                as_tuple=False
            )
        elif "roberta-base" in self.model_name:
            individual_subword_indices = (offset_mapping[:, :, 0] > 1).nonzero(
                as_tuple=False
            )
        if individual_subword_indices.nelement() != 0:
            grouped_subword_indices = []
            index_group = None
            for i in range(len(individual_subword_indices)):
                if index_group == None:
                    index_group = [
                        torch.tensor(
                            [
                                individual_subword_indices[i][0],
                                individual_subword_indices[i][1] - 1,
                            ],
                            device=self.device,
                        ),
                        individual_subword_indices[i],
                    ]
                    continue

                if (index_group[-1][0] == (individual_subword_indices[i][0])) and (
                    index_group[-1][1] == (individual_subword_indices[i][1] - 1)
                ):
                    index_group.append(
                        torch.tensor(
                            [
                                individual_subword_indices[i][0],
                                individual_subword_indices[i][1],
                            ],
                            device=self.device,
                        )
                    )

                else:
                    grouped_subword_indices.append(torch.stack(index_group))
                    index_group = [
                        torch.tensor(
                            [
                                individual_subword_indices[i][0],
                                individual_subword_indices[i][1] - 1,
                            ],
                            device=self.device,
                        ),
                        individual_subword_indices[i],
                    ]
            grouped_subword_indices.append(torch.stack(index_group))

            for group in grouped_subword_indices:
                for index in group:
                    index[1] -= 1

            for group in grouped_subword_indices:
                replacement_index_i = group[0][0]
                replacement_index_j = group[0][1]
                subword_indices = torch.stack([subword[1] for subword in group])
                subword_values = torch.index_select(
                    torch.index_select(
                        token_attention_output, dim=0, index=replacement_index_i,
                    ),
                    dim=1,
                    index=subword_indices,
                )
                if subword_method == "max":
                    replacement = torch.max(subword_values)
                elif subword_method == "mean":
                    replacement = torch.mean(subword_values)
                word_attention_output[
                    replacement_index_i, replacement_index_j
                ] = replacement

        return word_attention_output


class SeqClassModel(PreTrainedModel):
    model_name: str
    config_dict: Dict
    base_model_prefix = "seq_class"

    def __init__(self, model_config, config_dict):
        super().__init__(model_config)

        self.config_dict = config_dict
        self.model_name = self.config_dict.get("model_name")

        set_seed(self.config_dict["seed"])

        self.num_labels = model_config.num_labels
        self.initializer_name = self.config_dict["initializer_name"]

        self.bert = AutoModel.from_pretrained(
            self.config_dict["model_name"],
            from_tf=bool(".ckpt" in self.config_dict["model_name"]),
            config=model_config,
        )

        cnt = 0
        for layer in self.bert.children():
            if cnt >= self.config_dict.get("freeze_bert_layers_up_to", 0):
                break
            cnt += 1
            for param in layer.parameters():
                param.requires_grad = False

        self.post_bert_model = None
        if self.config_dict["soft_attention"]:
            self.post_bert_model = SoftAttentionSeqClassModel(
                self.config_dict, self.bert.config.hidden_size, model_config.num_labels,
            )
        else:
            self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
            self.classifier = nn.Linear(
                model_config.hidden_size, model_config.num_labels
            )
            self.dropout.apply(self.init_weights)
            self.classifier.apply(self.init_weights)

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def init_weights(self, m):
        if self.initializer_name == "normal":
            self.initializer = nn.init.normal_
        elif self.initializer_name == "glorot":
            self.initializer = nn.init.xavier_normal_
        elif self.initializer_name == "xavier":
            self.initializer = nn.init.xavier_uniform_

        if isinstance(m, nn.Linear):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        token_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_scores=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # last hidden states, pooler output, hidden states, attentions
        if self.post_bert_model is not None:
            outputs = self.post_bert_model.forward(
                outputs,
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                token_labels=token_labels,
                token_scores=token_scores,
                **kwargs,
            )  # (loss), logits, word attentions
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            # logits shape: (batch_size, num_labels), labels shape: (batch_size)
            loss = None
            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

