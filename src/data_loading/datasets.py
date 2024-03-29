import os
import csv
import numpy as np
import sys
import random

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
        token_label_mode="first",
        conll_10_type="cue",
        wi_locness_type="A",
        include_special_tokens=False,
        use_lowercase=False,
        max_sequence_length=512,
        shuffle=True,
    ):
        datasets = ["fce", "conll_10", "toxic", "wi_locness"]

        self.tokenizer = tokenizer
        self.mode = mode
        self.token_label_mode = token_label_mode
        self.include_special_tokens = include_special_tokens
        self.use_lowercase = use_lowercase
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle

        if dataset_name == "conll_10":
            data_dir = os.path.join(root_dir, "data", "external", "conll_10")
            tsv_file = os.path.join(
                data_dir, "conll_10_{}_{}.tsv".format(conll_10_type, mode)
            )
            self.positive_label = "C"
            self.negative_label = "O"

        elif dataset_name == "fce":
            data_dir = os.path.join(root_dir, "data", "processed", "fce_v2.1")
            tsv_file = os.path.join(data_dir, "fce_{}.tsv".format(mode))
            self.positive_label = "i"
            self.negative_label = "c"

        elif dataset_name == "toxic":
            data_dir = os.path.join(root_dir, "data", "processed", "toxic")
            tsv_file = os.path.join(data_dir, "tsd_{}.tsv".format(mode))
            self.positive_label = "t"
            self.negative_label = "f"

        elif dataset_name == "wi_locness":
            data_dir = os.path.join(root_dir, "data", "processed", "wi_locness_v2.1")
            tsv_file = os.path.join(
                data_dir, "wi_locness_{}_{}.tsv".format(wi_locness_type, mode)
            )
            self.positive_label = "i"
            self.negative_label = "c"

        else:
            raise Exception(
                "No valid dataset selected, please use one of: {}".format(datasets)
            )

        (
            self.samples,
            self.sentence_label,
            self.token_labels,
        ) = self._get_samples_labels(tsv_file)

        self.shuffle_indices = list(range(len(self.samples)))
        random.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.shuffle_indices[idx]
        sequence = self.samples[idx]
        sentence_label = self.sentence_label[idx]
        token_labels = self.token_labels[idx]

        return sequence, sentence_label, token_labels

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
                    if self.use_lowercase:
                        words.append(row[0].lower())
                    else:
                        words.append(row[0])
                    label = 1 if row[1] == self.positive_label else 0
                    word_labels.append(label)

                else:
                    if words:
                        tokenized_words, tokenized_word_labels = self._tokenize_input(
                            words, word_labels, self.token_label_mode
                        )

                        sentence_label = max(tokenized_word_labels)
                        samples.append(words)
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
                        encoded_labels.append(-2)

        encoded_output = [word for sublist in encoded_tokens for word in sublist]

        encoded_labels = encoded_labels[: (self.max_sequence_length - 2)]

        if self.include_special_tokens:
            encoded_labels.insert(0, encoded_labels)
            encoded_labels.append(encoded_labels)
        else:
            encoded_labels.insert(0, -3)
            encoded_labels.append(-1)

        return encoded_output, encoded_labels

