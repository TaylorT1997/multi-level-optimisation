import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer


class BinaryTSVDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        root_dir="/home/tom/Projects/multi-level-optimisation/",
        mode="dev",
        conll_10_type="cue",
        wi_locness_type="A",
    ):
        datasets = ["fce", "conll_10", "toxic", "wi_locness"]

        self.tokenizer = tokenizer
        self.mode = mode

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

        self.samples, self.labels = self._get_samples_labels(tsv_file)
        self.max_length = self._get_max_length(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence = self.samples[idx]
        label = self.labels[idx]

        encoded_sequence = self.tokenizer.encode_plus(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return encoded_sequence, label

    def _get_samples_labels(self, tsv_file):
        samples = []
        sentence_labels = []

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
                        tokenized_words = self._tokenize_input(words)

                        samples.append(tokenized_words)
                        sentence_labels.append(sentence_label)

                    words = []
                    word_labels = []
                    continue

        return samples, sentence_labels

    def _tokenize_input(self, words):
        encoded_words = []

        for word in words:
            encoded_word = self.tokenizer.encode(word, add_special_tokens=False)
            encoded_words.append(encoded_word)

        encoded_output = [word for sublist in encoded_words for word in sublist]
        return encoded_output

    def _add_masked_tokens(self):
        pass

    def _get_max_length(self, sentences):
        max_length = 0
        for sentence in sentences:
            length = len(sentence)
            if length > max_length:
                max_length = length
        return max_length
