import enum
import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel,
    DebertaConfig,
    DebertaModel,
    RobertaConfig,
    RobertaModel,
)
import sys


def bmul(vec, mat, axis=0):
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


class TokenModel(nn.Module):
    def __init__(
        self,
        pretrained_model="bert-base-cased",
        soft_attention_beta=1,
        sentence_loss_weight=1,
        token_loss_weight=1,
        regularizer_loss_weight=0.01,
        token_supervision=True,
        sequence_supervision=True,
        regularization_losses=True,
        normalise_supervised_losses=False,
        normalise_regularization_losses=False,
        subword_method="max",
        device="cuda",
        debug=False,
    ):
        super(TokenModel, self).__init__()
        if "bert-base" in pretrained_model:
            model_config = BertConfig.from_pretrained(pretrained_model, num_labels=1)
            self.seq2seq_model = BertModel(model_config)
        elif "deberta-base" in pretrained_model:
            model_config = DebertaConfig.from_pretrained(pretrained_model, num_labels=1)
            self.seq2seq_model = DebertaModel(model_config)
        elif "roberta-base" in pretrained_model:
            model_config = RobertaConfig.from_pretrained(pretrained_model, num_labels=1)
            self.seq2seq_model = RobertaModel(model_config)
        else:
            print(f"Model {pretrained_model} not in library")

        self.token_attention = nn.Sequential(
            nn.Linear(self.seq2seq_model.config.hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

        self.sentence_classification = nn.Sequential(
            nn.Linear(self.seq2seq_model.config.hidden_size, 300),
            nn.Tanh(),
            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

        # Apply weight initialisation
        self.token_attention.apply(self._init_weights)
        self.sentence_classification.apply(self._init_weights)

        self.soft_attention_beta = soft_attention_beta
        self.sentence_loss_weight = sentence_loss_weight
        self.token_loss_weight = token_loss_weight
        self.regularizer_loss_weight = regularizer_loss_weight
        self.normalise_supervised_losses = normalise_supervised_losses
        self.normalise_regularization_losses = normalise_regularization_losses
        self.token_supervision = token_supervision
        self.sequence_supervision = sequence_supervision
        self.regularization_losses = regularization_losses
        self.subword_method = subword_method

        self.device = device
        self.debug = debug

        self.step = 0

    def forward(self, input_ids, attention_mask=None, offset_mapping=None, labels=None):

        # Pass tokens through pretrained model
        pretrained_output = self.seq2seq_model(input_ids, attention_mask)

        if self.debug:
            print(
                f"pretrained_output_last_hidden_state shape: \n{pretrained_output.last_hidden_state.shape}\n"
            )
            print(
                f"pretrained_output_last_hidden_state: \n{pretrained_output.last_hidden_state}\n"
            )

        # Pass pretrained output through attention layer
        token_attention_output = self.token_attention(
            pretrained_output.last_hidden_state
        ).squeeze(2)

        if self.debug:
            print(f"token_attention_output shape: \n{token_attention_output.shape}\n")
            print(f"token_attention_output: \n{token_attention_output}\n")

        word_attention_output = token_attention_output.clone()
        individual_subword_indices = (offset_mapping[:, :, 0] != 0).nonzero(
            as_tuple=False
        )

        if individual_subword_indices.nelement() != 0 and self.subword_method != "first":
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
                    index_group.append(individual_subword_indices[i])

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

        if self.debug:
            print(f"word_attention_output shape: \n{word_attention_output.shape}\n")
            print(f"word_attention_output: \n{word_attention_output}\n")

        # Mask padded tokens and subword tokens
        token_attention_mask = torch.where(
            ((labels == 0) | (labels == 1)),
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )

        not_subword = offset_mapping[:, :, 0] == 0
        subword_attention_mask = torch.where(
            not_subword, torch.ones_like(not_subword), torch.zeros_like(not_subword)
        )
        token_attention_mask = torch.mul(token_attention_mask, subword_attention_mask)

        masked_token_attention_output = torch.mul(
            word_attention_output, token_attention_mask
        )

        if self.debug:
            print(
                f"masked_token_attention_output shape: \n{masked_token_attention_output.shape}\n"
            )
            print(f"masked_token_attention_output: \n{masked_token_attention_output}\n")

        # Normalise the attention output
        token_attention_output_normalised = torch.pow(
            masked_token_attention_output, self.soft_attention_beta
        ) / torch.sum(
            torch.pow(masked_token_attention_output, self.soft_attention_beta),
            dim=1,
            keepdim=True,
        )

        if self.debug:
            print(
                f"token_attention_output_normalised shape: \n{token_attention_output_normalised.shape}\n"
            )
            print(
                f"token_attention_output_normalised: \n{token_attention_output_normalised}\n"
            )

        # Apply normalised attention to pretrained output
        pretrained_output_with_attention = torch.bmm(
            pretrained_output.last_hidden_state.transpose(1, 2),
            token_attention_output_normalised.unsqueeze(2),
        ).squeeze(2)

        if self.debug:
            print(
                f"pretrained_output_with_attention shape: \n{pretrained_output_with_attention.shape}\n"
            )
            print(
                f"pretrained_output_with_attention : \n{pretrained_output_with_attention}\n"
            )

        # token_attention_output_normalised_expanded = token_attention_output_normalised.unsqueeze(
        #     2
        # ).expand(
        #     -1, -1, 768
        # )

        # if self.debug:
        #     print(
        #         f"token_attention_output_normalised_expanded shape: \n{token_attention_output_normalised_expanded.shape}\n"
        #     )
        #     print(
        #         f"token_attention_output_normalised_expanded: \n{token_attention_output_normalised_expanded}\n"
        #     )

        # pretrained_output_with_attention = torch.einsum(
        #     "bij, bij -> bij",
        #     pretrained_output.last_hidden_state,
        #     token_attention_output_normalised_expanded,
        # )

        # pretrained_output_with_attention_summed = torch.sum(
        #     pretrained_output_with_attention, dim=1
        # )

        # if self.debug:
        #     print(
        #         f"pretrained_output_with_attention_summed shape: \n{pretrained_output_with_attention_summed.shape}\n"
        #     )
        #     print(
        #         f"pretrained_output_with_attention_summed : \n{pretrained_output_with_attention_summed}\n"
        #     )

        # Pass pretrained output with attention through sentence classifiaction layer
        sentence_classification_output = self.sentence_classification(
            pretrained_output_with_attention
        )

        if self.debug:
            print(
                f"sentence_classification_output shape: \n{sentence_classification_output.shape}\n"
            )
            print(
                f"sentence_classification_output: \n{sentence_classification_output}\n"
            )

        # Calculate the model loss
        (
            model_loss,
            sentence_loss,
            token_loss,
            regularizer_loss_a,
            regularizer_loss_b,
        ) = self._calculate_loss(
            masked_token_attention_output, sentence_classification_output, labels,
        )

        output = {
            "loss": model_loss,
            "sentence_loss": sentence_loss,
            "token_loss": token_loss,
            "regularizer_loss_a": regularizer_loss_a,
            "regularizer_loss_b": regularizer_loss_b,
            "token_embeddings": pretrained_output.last_hidden_state,
            "token_logits": masked_token_attention_output,
            "sequence_logits": sentence_classification_output,
        }

        return output

    def _calculate_loss(
        self, token_attention_output, sentence_classification_output, labels,
    ):
        if self.debug:
            print(f"token_attention_output: \n{token_attention_output}\n")
            print(f"token_attention_output shape: \n{token_attention_output.shape}\n")
            print(f"labels: \n{labels}\n")
            print(f"labels shape: \n{labels.shape}\n")

        # Get batch size
        batch_size = labels.shape[0]

        # Get the sentence label from the token labels
        sentence_labels = torch.max(labels, dim=1, keepdim=True).values

        if self.debug:
            print(f"sentence_labels: \n{sentence_labels}\n")
            print(f"sentence_labels shape: \n{sentence_labels.shape}\n")

        # Calculate the sentence MSE loss
        mse_loss = nn.MSELoss(reduction="sum")
        sentence_loss = mse_loss(sentence_classification_output, sentence_labels)

        # Calculate the token MSE loss depending on the subword method
        mse_loss = nn.MSELoss(reduction="sum")
        zero_labels = torch.where(labels == 1, labels, torch.zeros_like(labels))
        token_loss = mse_loss(token_attention_output, zero_labels)

        if self.debug:
            print(f"zero_labels: \n{zero_labels}\n")
            print(f"zero_labels shape: \n{zero_labels.shape}\n")

        # Normalise the MSE losses (optionally)+
        if self.normalise_supervised_losses:
            sentence_loss = sentence_loss / batch_size
            token_loss = token_loss / batch_size

        # Create token attention tensor with mask of ones instead of zeros
        token_attention_output_ones_mask = torch.where(
            token_attention_output == 0,
            torch.ones_like(token_attention_output),
            token_attention_output,
        )

        # Calculate regularisation losses
        regularizer_loss_a = torch.sum(
            torch.pow(
                torch.min(token_attention_output_ones_mask, dim=1, keepdim=True).values
                - 0,
                2,
            )
        )
        regularizer_loss_b = torch.sum(
            torch.pow(
                torch.max(token_attention_output, dim=1, keepdim=True,).values
                - sentence_labels,
                2,
            )
        )

        if self.debug:
            print(f"sentence_loss: \n{sentence_loss}\n")
            print(f"token_loss: \n{token_loss}\n")
            print(f"regularizer_loss_a: \n{regularizer_loss_a}\n")
            print(f"regularizer_loss_b: \n{regularizer_loss_b}\n")

        # Combine regularize losses and (optionally) normalise
        if self.normalise_regularization_losses:
            regularizer_loss_a = regularizer_loss_a / batch_size
            regularizer_loss_b = regularizer_loss_b / batch_size
        regularizer_losses = regularizer_loss_a + regularizer_loss_b

        # Apply loss weights
        if self.token_supervision:
            token_loss_weighted = token_loss * self.token_loss_weight
        else:
            token_loss_weighted = 0
        if self.sequence_supervision:
            sentence_loss_weighted = sentence_loss * self.sentence_loss_weight
        else:
            sentence_loss_weighted = 0
        if self.regularization_losses:
            regularizer_loss_weighted = (
                regularizer_losses * self.regularizer_loss_weight
            )
        else:
            regularizer_loss_weighted = 0

        # Calculate total loss
        total_loss = (
            sentence_loss_weighted + token_loss_weighted + regularizer_loss_weighted
        )

        if self.debug:
            print(f"total_loss: \n{total_loss}\n")

        return (
            total_loss,
            sentence_loss,
            token_loss,
            regularizer_loss_a,
            regularizer_loss_b,
        )

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
