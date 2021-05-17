import os
import csv
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import BertTokenizer

class BinaryTokenTSVDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        root_dir="/home/tom/Projects/multi-level-optimisation/",
        mode="train",
        token_label_mode="all",
        conll_10_type="cue",
        wi_locness_type="A",
        include_special_tokens=False,
        max_length=None,
    ):
        datasets = ["fce", "conll_10", "toxic", "wi_locness"]

        self.tokenizer = tokenizer
        self.mode = mode
        self.token_label_mode = token_label_mode
        self.include_special_tokens = include_special_tokens

        if dataset_name == "conll_10":
            data_dir = os.path.join(root_dir, "data", "processed", "conll_10")
            tsv_file = os.path.join(
                data_dir, "conll_10_{}_{}.tsv".format(conll_10_type, mode)
            )
            self.positive_label = "C"

        elif dataset_name == "fce":
            data_dir = os.path.join(root_dir, "data", "processed", "fce_v2.1")
            tsv_file = os.path.join(data_dir, "fce_{}.tsv".format(mode))
            self.positive_label = "i"

        elif dataset_name == "toxic":
            data_dir = os.path.join(root_dir, "data", "processed", "toxic")
            tsv_file = os.path.join(data_dir, "tsd_{}.tsv".format(mode))
            self.positive_label = "t"

        elif dataset_name == "wi_locness":
            data_dir = os.path.join(root_dir, "data", "processed", "wi_locness_v2.1")
            tsv_file = os.path.join(
                data_dir, "wi_locness_{}_{}.tsv".format(wi_locness_type, mode)
            )
            self.positive_label = "i"

        else:
            raise Exception(
                "No valid dataset selected, please use one of: {}".format(datasets)
            )

        (
            self.samples,
            self.sentence_label,
            self.token_labels,
        ) = self._get_samples_labels(tsv_file)

        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = self._get_max_length(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence = self.samples[idx]
        sentence_label = self.sentence_label[idx]
        token_labels = self.token_labels[idx]

        encoded_sequence = self.tokenizer.encode_plus(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.include_special_tokens:
            token_labels.insert(0, sentence_label)
            token_labels.append(sentence_label)
            token_labels.extend([-1] * (self.max_length - len(token_labels)))
            token_labels = token_labels[:self.max_length]
        else:
            token_labels.insert(0, -1)
            token_labels.extend([-1] * (self.max_length - len(token_labels)))
            token_labels = token_labels[:self.max_length]

        return encoded_sequence, sentence_label, token_labels

    def _get_samples_labels(self, tsv_file):
        samples = []
        sentence_labels = []
        token_labels = []

        with open(tsv_file, "r", encoding="utf-8") as tsv:
            tsv_reader = csv.reader(tsv, delimiter="\t")

            words = []
            word_labels = []

            for row in tsv_reader:
                if row:
                    words.append(row[0])
                    label = 1 if row[1] == self.positive_label else 0
                    word_labels.append(label)

                else:
                    if words:
                        sentence_label = max(word_labels)
                        tokenized_words, tokenized_word_labels = self._tokenize_input(
                            words, word_labels, self.token_label_mode
                        )

                        samples.append(tokenized_words)
                        sentence_labels.append(sentence_label)
                        token_labels.append(tokenized_word_labels)

                    words = []
                    word_labels = []
                    continue

        return samples, sentence_labels, token_labels

    def _tokenize_input(self, words, word_labels, mode="all"):
        encoded_tokens = []
        encoded_labels = []

        for i in range(len(words)):
            encoded_word = self.tokenizer.encode(words[i], add_special_tokens=False)
            encoded_tokens.append(encoded_word)

            num_word_parts = len(encoded_word)

            if mode == "all":
                for j in range(num_word_parts):
                    encoded_labels.append(word_labels[i])
            elif mode == "first":
                for j in range(num_word_parts):
                    if j == 0:
                        encoded_labels.append(word_labels[i])
                    else:
                        encoded_labels.append(abs(word_labels[i] - 1))

        encoded_output = [word for sublist in encoded_tokens for word in sublist]
        return encoded_output, encoded_labels

    def _get_max_length(self, sentences):
        max_length = 0
        for sentence in sentences:
            length = len(sentence)
            if length > max_length:
                max_length = length
        return min(max_length + 2, 512)

