import dataclasses
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
    DebertaModel,
    BertPreTrainedModel,
    BertConfig,
    RobertaConfig,
    DebertaConfig,
    set_seed,
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


class SoftAttentionSeqClassModel(nn.Module):
    def __init__(
        self,
        pretrained_model="bert-base-cased",
        use_attention_layer=True,
        num_labels=2,
        soft_attention_beta=1,
        sentence_loss_weight=1,
        token_loss_weight=1,
        regularizer_loss_weight=0.01,
        dropout=0.1,
        token_supervision=True,
        sequence_supervision=True,
        regularization_losses=True,
        normalise_supervised_losses=False,
        normalise_regularization_losses=False,
        use_sequence_layer=True,
        subword_method="max",
        mask_subwords=False,
        initializer_name="glorot",
        bert_out_size=768,
        seed=666,
        debug=False,
    ):
        super(SoftAttentionSeqClassModel, self).__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.pretrained_model = pretrained_model
        self.use_attention_layer = use_attention_layer
        self.num_labels = num_labels
        self.soft_attention_beta = soft_attention_beta
        self.sentence_loss_weight = sentence_loss_weight
        self.token_loss_weight = token_loss_weight
        self.regularizer_loss_weight = regularizer_loss_weight
        self.normalise_supervised_losses = normalise_supervised_losses
        self.normalise_regularization_losses = normalise_regularization_losses
        self.dropout = dropout
        self.token_supervision = token_supervision
        self.sequence_supervision = sequence_supervision
        self.regularization_losses = regularization_losses
        self.subword_method = subword_method
        self.mask_subwords = mask_subwords
        self.initializer_name = initializer_name

        self.debug = debug

        self.use_sequence_layer = use_sequence_layer

        self.dropout = nn.Dropout(p=self.dropout)

        self.attention_evidence = nn.Linear(bert_out_size, 100)
        self.attention_weights = nn.Linear(100, 1)

        self.final_hidden = nn.Linear(bert_out_size, 300,)
        self.result_layer = nn.Linear(300, 2)

        self.pretrained_model = pretrained_model
        self.attention_act = torch.sigmoid

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
        bert_hidden_outputs,
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
        inp_lengths = (attention_mask != 0).sum(dim=1) - 1  # exc CLS removed next
        bert_length = input_ids.shape[1] - 1
        bert_hidden_outputs = bert_hidden_outputs[:, 1:]  # remove CLS
        after_dropout = self.dropout(bert_hidden_outputs)
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
        attn_weights = torch.pow(attn_weights, self.soft_attention_beta)

        if self.debug:
            print(f"attn_weights_squared shape: \n{attn_weights.shape}\n")
            print(f"attn_weights_squared: \n{attn_weights}\n")

        # normalise attn weights
        attn_weights = attn_weights / torch.sum(attn_weights, dim=1, keepdim=True)
        self.attention_weights_normalised = attn_weights

        if self.debug:
            print(f"attn_weights_normalised shape: \n{attn_weights.shape}\n")
            print(f"attn_weights_normalised: \n{attn_weights}\n")

        proc_tensor = torch.bmm(
            after_dropout.transpose(1, 2), attn_weights.unsqueeze(2)
        ).squeeze(dim=2)
        proc_tensor = torch.tanh(self.final_hidden(proc_tensor))

        self.sentence_scores = torch.sigmoid(self.result_layer(proc_tensor))
        self.sentence_scores = self.sentence_scores.view(
            [bert_hidden_outputs.shape[0], self.num_labels]
        )

        if self.debug:
            print(
                f"attention_weights_unnormalised shape: \n{self.attention_weights_unnormalised.shape}\n"
            )
            print(
                f"attention_weights_unnormalised: \n{self.attention_weights_unnormalised}\n"
            )

        if not self.use_sequence_layer:
            print("!" * 50)
            max_token_attention = torch.max(self.attention_weights_unnormalised, dim=1)
            self.sentence_scores = max_token_attention.values.unsqueeze(1)

        if self.debug:
            print(f"proc_tensor shape: \n{proc_tensor.shape}\n")
            print(f"proc_tensor: \n{proc_tensor}\n")

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
        loss = None
        if labels is not None:
            # SEQUENCE LOSS
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = self.sentence_loss_weight * loss_fct(
                    self.sentence_scores.view(-1), labels.view(-1)
                )
            else:
                loss_fct = CrossEntropyLoss()
                loss = self.sentence_loss_weight * loss_fct(
                    self.sentence_scores.view(-1, self.num_labels), labels.view(-1)
                )

            if self.debug:
                print(f"num_labels: \n{self.num_labels}\n")
                print(f"labels: \n{labels}\n")
                print(f"loss: \n{loss}\n")
                print(self.sentence_scores.shape[1])

            # REGULARIZER LOSS A
            if self.regularizer_loss_weight != 0:
                min_attentions, _ = torch.min(
                    torch.where(
                        self._sequence_mask(inp_lengths, maxlen=bert_length),
                        self.attention_weights_unnormalised,
                        torch.zeros_like(self.attention_weights_unnormalised) + 1e6,
                    ),  # [:, 1:],
                    dim=1,
                )
                l2 = self.regularizer_loss_weight * torch.mean(
                    torch.square(min_attentions.view(-1))
                )
                loss += l2

            if self.debug:
                print(f"alpha min_attentions: \n{min_attentions}\n")
                print(f"alpha l2: \n{l2}\n")

            # REGULARIZER LOSS B
            if self.regularizer_loss_weight != 0:
                attn_weights_masked = torch.where(
                    self._sequence_mask(inp_lengths, maxlen=bert_length),
                    self.attention_weights_unnormalised,
                    torch.zeros_like(self.attention_weights_unnormalised) - 1e6,
                )
                max_attentions, _ = torch.max(attn_weights_masked, dim=1,)
                l3 = self.regularizer_loss_weight * torch.mean(
                    torch.square(max_attentions.view(-1) - labels.view(-1))
                )
                loss += l3

            if self.debug:
                print(f"gamma attn_weights_masked: \n{attn_weights_masked}\n")
                print(f"gamma max_attentions: \n{max_attentions}\n")
                print(f"gamma l3: \n{l3}\n")

                print(f"token_labels: \n{token_labels}\n")

            # TOKEN LOSS
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

                masked_word_attention = torch.where(
                    ((token_labels == 0) | (token_labels == 1)),
                    word_attentions,
                    torch.zeros_like(token_labels),
                )

                loss += self.token_loss_weight * token_loss

                if self.debug:
                    print(f"zero_labels: \n{zero_labels}\n")
                    print(f"masked_token_attention: \n{masked_token_attention}\n")
                    print(f"token_loss: \n{token_loss}\n")

            outputs = (loss,) + outputs

            if self.debug:
                print(f"outputs: \n{outputs}\n")

        if self.debug:
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
        self, token_attention_output, offset_mapping,
    ):
        word_attention_output = token_attention_output.clone()

        if "bert-base" in self.pretrained_model:
            individual_subword_indices = (offset_mapping[:, :, 0] != 0).nonzero(
                as_tuple=False
            )
        elif "deberta-base" in self.pretrained_model:
            individual_subword_indices = (offset_mapping[:, :, 0] != 0).nonzero(
                as_tuple=False
            )
        elif "roberta-base" in self.pretrained_model:
            individual_subword_indices = (offset_mapping[:, :, 0] > 1).nonzero(
                as_tuple=False
            )
        if individual_subword_indices.nelement() != 0:
            # print(individual_subword_indices)
            grouped_subword_indices = []
            index_group = None
            for i in range(len(individual_subword_indices)):
                # print(individual_subword_indices[i])
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
                    # print(index_group)
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

            # print(grouped_subword_indices)

            for group in grouped_subword_indices:
                for index in group:
                    index[1] -= 1

            # print(grouped_subword_indices)

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
                if self.subword_method == "max":
                    replacement = torch.max(subword_values)
                elif self.subword_method == "mean":
                    replacement = torch.mean(subword_values)
                word_attention_output[
                    replacement_index_i, replacement_index_j
                ] = replacement

        return word_attention_output


class SeqClassModel(PreTrainedModel):
    def __init__(
        self,
        model_config,
        pretrained_model="bert-base-cased",
        use_attention_layer=True,
        num_labels=2,
        soft_attention_beta=1,
        sentence_loss_weight=1,
        token_loss_weight=1,
        regularizer_loss_weight=0.01,
        dropout=0.1,
        token_supervision=True,
        sequence_supervision=True,
        regularization_losses=True,
        normalise_supervised_losses=False,
        normalise_regularization_losses=False,
        use_sequence_layer=True,
        subword_method="max",
        mask_subwords=False,
        initializer_name="glorot",
        seed=666,
        debug=False,
    ):
        super().__init__(model_config)

        self.pretrained_model = pretrained_model
        self.use_attention_layer = use_attention_layer
        self.num_labels = num_labels
        self.soft_attention_beta = soft_attention_beta
        self.sentence_loss_weight = sentence_loss_weight
        self.token_loss_weight = token_loss_weight
        self.regularizer_loss_weight = regularizer_loss_weight
        self.normalise_supervised_losses = normalise_supervised_losses
        self.normalise_regularization_losses = normalise_regularization_losses
        self.dropout = dropout
        self.token_supervision = token_supervision
        self.sequence_supervision = sequence_supervision
        self.regularization_losses = regularization_losses
        self.subword_method = subword_method
        self.mask_subwords = mask_subwords
        self.initializer_name = initializer_name
        self.use_sequence_layer = use_sequence_layer

        self.debug = debug
        self.pretrained_model = pretrained_model
        self.seed = seed

        set_seed(self.seed)

        self.bert = AutoModel.from_pretrained(
            self.pretrained_model,
            from_tf=bool(".ckpt" in self.pretrained_model),
            config=model_config,
        )

        if "bert-base" in pretrained_model:
            model_config = BertConfig.from_pretrained(
                pretrained_model, num_labels=self.num_labels
            )
            self.bert = BertModel(model_config)
        elif "deberta-base" in pretrained_model:
            model_config = DebertaConfig.from_pretrained(
                pretrained_model, num_labels=self.num_labels
            )
            self.bert = DebertaModel(model_config)
        elif "roberta-base" in pretrained_model:
            model_config = RobertaConfig.from_pretrained(
                pretrained_model, num_labels=self.num_labels
            )
            self.bert = RobertaModel(model_config)
        else:
            print(f"Model {pretrained_model} not in library")

        self.post_bert_model = None
        if self.use_attention_layer:
            self.post_bert_model = SoftAttentionSeqClassModel(
                pretrained_model=self.pretrained_model,
                use_attention_layer=self.use_attention_layer,
                num_labels=model_config.num_labels,
                soft_attention_beta=self.soft_attention_beta,
                sentence_loss_weight=self.soft_attention_beta,
                token_loss_weight=self.token_loss_weight,
                regularizer_loss_weight=self.regularizer_loss_weight,
                dropout=self.dropout,
                token_supervision=self.token_supervision,
                sequence_supervision=self.sequence_supervision,
                regularization_losses=self.regularization_losses,
                normalise_supervised_losses=self.normalise_supervised_losses,
                normalise_regularization_losses=self.normalise_regularization_losses,
                use_sequence_layer=self.use_sequence_layer,
                subword_method=self.subword_method,
                mask_subwords=self.mask_subwords,
                initializer_name=self.initializer_name,
                seed=self.seed,
                debug=self.debug,
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
                outputs[0],
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

