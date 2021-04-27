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
        freeze_pretrained_model=False,
        sentence_loss_weight=1,
        token_loss_weight=1,
        token_supervision=True,
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

        self.sentence_loss_weight = sentence_loss_weight
        self.token_loss_weight = token_loss_weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass tokens through pretrained model
        pretrained_output = self.seq2seq_model(input_ids, attention_mask)

        print(
            "prerained_output_last_hidden_state shape: {}".format(
                pretrained_output.last_hidden_state.shape
            )
        )

        # Pass pretrained output through attention layer
        token_attention_output = self.token_attention(
            pretrained_output.last_hidden_state
        )

        print("token_attention_output shape: {}".format(token_attention_output.shape))

        # Normalise the attention output
        token_attention_output_normalised = token_attention_output / torch.sum(
            token_attention_output, dim=1, keepdim=True
        )

        print(
            "token_attention_output_normalised shape: {}".format(
                token_attention_output_normalised.shape
            )
        )

        # Apply normalised attention to pretrained output
        pretrained_output_with_attention = torch.matmul(
            pretrained_output.last_hidden_state.transpose(1, 2),
            token_attention_output_normalised,
        ).squeeze(dim=2)

        print(
            "pretrained_output_with_attention shape: {}".format(
                pretrained_output_with_attention.shape
            )
        )

        # Pass pretrained output with attention through sentence classifiaction layer
        sentence_classification_output = self.sentence_classification(
            pretrained_output_with_attention
        )

        print(
            "sentence_classification_output shape: {}".format(
                sentence_classification_output.shape
            )
        )

        # Bring together all of the relevant model outputs
        model_outputs = {
            "pretrained_output": pretrained_output.last_hidden_state,
            "token_attention_output": token_attention_output,
            "sentence_classification_output": sentence_classification_output,
        }

        # Calculate the model loss
        model_loss = self._calculate_loss(
            token_attention_output, sentence_classification_output, labels
        )

        output = {
            "loss": model_loss,
            "token_logits": token_attention_output.squeeze(2),
            "sequence_logits": sentence_classification_output,
        }

        return output

    def _calculate_loss(
        self, token_attention_output, sentence_classification_output, labels
    ):
        # Get the sentence label from the token labels
        sentence_labels = torch.max(labels)

        # Calculate the sentence MSE loss
        mse_loss = nn.MSELoss()
        sentence_loss = mse_loss(sentence_classification_output, sentence_labels)

        # Calculate the token MSE loss
        mse_loss = nn.MSELoss(reduction="sum")
        token_loss = mse_loss(token_attention_output.squeeze(2), labels)

        # Apply loss weights
        sentence_loss_weighted = sentence_loss * self.sentence_loss_weight
        token_loss_weighted = token_loss * self.token_loss_weight

        # Calculate total loss
        total_loss = sentence_loss_weighted + token_loss_weighted

        return total_loss
