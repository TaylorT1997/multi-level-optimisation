import os
import csv

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
            positive_label = "C"

        elif dataset_name == "fce":
            data_dir = os.path.join(root_dir, "data", "processed", "fce_v2.1")
            tsv_file = os.path.join(data_dir, "fce_{}.tsv".format(mode))
            positive_label = "i"

        elif dataset_name == "toxic":
            data_dir = os.path.join(root_dir, "data", "processed", "toxic")
            tsv_file = os.path.join(data_dir, "tsd_{}.tsv".format(mode))
            positive_label = "t"

        elif dataset_name == "wi_locness":
            data_dir = os.path.join(root_dir, "data", "processed", "wi_locness_v2.1")
            tsv_file = os.path.join(
                data_dir, "wi_locness_{}_{}.tsv".format(wi_locness_type, mode)
            )
            positive_label = "i"

        else:
            raise Exception(
                "No valid dataset selected, please use one of: {}".format(datasets)
            )

        self._get_samples(tsv_file)

    def __len__(self):
        return None

    def __get_item__(self, idx):
        return None

    def _get_samples(self, tsv_file):
        samples = []

        with open(tsv_file, "r", encoding="utf-8") as tsv:
            tsv_reader = csv.reader(tsv, delimiter="\t")

            words = []
            labels = []

            for row in tsv_reader:
                if row:
                    words.append(row[0])
                    labels.append(row[1])
                else:
                    break

            print(words)
            print(labels)

            print()

    def _tokenize_input(self, words):

        for word in words:
            encoded_word = tokenizer.encode(word)

        return tokenized_words

    def _add_masked_tokens(self,):
        pass


# dataset = BinaryTSVDataset(dataset_name="wi_locness")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "yo [MASK]"

encoding = tokenizer.encode()
print(encoding)
