import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    DebertaConfig,
    DebertaModel,
)


class TokenModel(nn.Module):
    def __init__(
        self,
        pretrained_model="bert-base-cased",
        soft_attention_beta=1,
        sentence_loss_weight=1,
        token_loss_weight=1,
        regularizer_loss_weight=0.01,
        token_supervision=True,
        normalise_supervised_losses=False,
        normalise_regularization_losses=False,
        debug=False,
    ):
        super(TokenModel, self).__init__()
        if "bert-base" in pretrained_model:
            model_config = BertConfig.from_pretrained(pretrained_model, num_labels=1)
            self.seq2seq_model = BertModel(model_config)
        elif "deberta-base" in pretrained_model:
            model_config = DebertaConfig.from_pretrained(pretrained_model, num_labels=1)
            self.seq2seq_model = DebertaModel(model_config)
        else:
            print("Model {} not in library".format(pretrained_model))

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

        self.soft_attention_beta = soft_attention_beta
        self.sentence_loss_weight = sentence_loss_weight
        self.token_loss_weight = token_loss_weight
        self.regularizer_loss_weight = regularizer_loss_weight
        self.normalise_supervised_losses = normalise_supervised_losses
        self.normalise_regularization_losses = normalise_regularization_losses
        self.token_supervision = token_supervision

        self.debug = debug

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass tokens through pretrained model
        pretrained_output = self.seq2seq_model(input_ids, attention_mask)

        if self.debug:
            print(
                "pretrained_output_last_hidden_state shape: {}".format(
                    pretrained_output.last_hidden_state.shape
                )
            )

        # Pass pretrained output through attention layer
        token_attention_output = self.token_attention(
            pretrained_output.last_hidden_state
        ).squeeze()

        if self.debug:
            print(
                "token_attention_output shape: {}".format(token_attention_output.shape)
            )

        # Mask padded tokens
        token_attention_mask = torch.where(
            labels != -1, torch.ones_like(labels), torch.zeros_like(labels)
        )

        if self.debug:
            print(
                "token_attention_mask shape: {}".format(
                    token_attention_mask.shape
                )
            )

        masked_token_attention_output = torch.mul(
            token_attention_output, token_attention_mask
        )

        if self.debug:
            print(
                "masked_token_attention_output shape: {}".format(
                    masked_token_attention_output.shape
                )
            )

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
                "token_attention_output_normalised shape: {}".format(
                    token_attention_output_normalised.shape
                )
            )

        # Apply normalised attention to pretrained output
        pretrained_output_with_attention = torch.matmul(
            pretrained_output.last_hidden_state.transpose(1, 2),
            token_attention_output_normalised.unsqueeze(2),
        )

        if self.debug:
            print(
                "pretrained_output_with_attention shape: {}".format(
                    pretrained_output_with_attention.shape
                )
            )

        # Pass pretrained output with attention through sentence classifiaction layer
        sentence_classification_output = self.sentence_classification(
            pretrained_output_with_attention.squeeze()
        )

        if self.debug:
            print(
                "sentence_classification_output shape: {}\n".format(
                    sentence_classification_output.shape
                )
            )

        # Calculate the model loss
        model_loss = self._calculate_loss(
            masked_token_attention_output, sentence_classification_output, labels
        )

        output = {
            "loss": model_loss,
            "token_embeddings": pretrained_output.last_hidden_state,
            "token_logits": masked_token_attention_output,
            "sequence_logits": sentence_classification_output,
        }

        return output

    def _calculate_loss(
        self, token_attention_output, sentence_classification_output, labels
    ):
        if self.debug:
            print("token_attention_output: {}".format(token_attention_output))
            print(
                "sentence_classification_output: {}".format(
                    sentence_classification_output
                )
            )
            print("labels: {}".format(labels))
            print(
                "token_attention_output shape: {}".format(token_attention_output.shape)
            )
            print(
                "sentence_classification_output shape: {}".format(
                    sentence_classification_output.shape
                )
            )
            print("labels shape: {}".format(labels.shape))

        # Get batch size
        batch_size = labels.shape[0]

        # Get the sentence label from the token labels
        sentence_labels = torch.max(labels, dim=1, keepdim=True).values

        if self.debug:
            print("sentence_labels: {}".format(sentence_labels))
            print("sentence_labels shape: {}".format(sentence_labels.shape))

        # Calculate the sentence MSE loss
        mse_loss = nn.MSELoss(reduction="sum")
        sentence_loss = mse_loss(sentence_classification_output, sentence_labels)

        # Calculate the token MSE loss
        zero_labels = torch.where(labels != -1, labels, torch.zeros_like(labels))
        token_loss = mse_loss(token_attention_output, zero_labels)

        # Normalise the MSE losses (optionally)
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
            print(sentence_loss)
            print(token_loss)
            print(regularizer_loss_a)
            print(regularizer_loss_b)

        # Combine regularize losses and (optionally) normalise
        regularizer_losses = regularizer_loss_a + regularizer_loss_b
        if self.normalise_regularization_losses:
            regularizer_losses = regularizer_losses / batch_size

        # Apply loss weights
        sentence_loss_weighted = sentence_loss * self.sentence_loss_weight
        token_loss_weighted = token_loss * self.token_loss_weight
        regularizer_loss_weighted = regularizer_losses * self.regularizer_loss_weight

        # Calculate total loss
        if self.token_supervision:
            total_loss = (
                sentence_loss_weighted + token_loss_weighted + regularizer_loss_weighted
            )
        else:
            total_loss = sentence_loss_weighted + regularizer_loss_weighted

        return total_loss
