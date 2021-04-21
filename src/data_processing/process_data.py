import sys
import os
import argparse
import csv
import shutil
import xml
import xml.sax
import nltk, re, pprint
from nltk import word_tokenize
import random
import pandas as pd
import ast

nltk.download("punkt")


class XMLHandler(xml.sax.handler.ContentHandler):
    """
    Adapted from code by M. Rei
    """

    def __init__(self, filtr):
        self.filtr = filtr
        self.in_sentence = False
        self.in_cue = 0
        self.in_xcope = 0
        self.sentence = []
        self.data = []

    def parse(self, f):
        xml.sax.parse(f, self)
        return self.data

    def characters(self, data):
        if self.in_sentence == False:
            return
        data = data.strip()
        if len(data) > 0:
            for token in word_tokenize(data):
                if self.filtr in ["cue", "cuescope"] and self.in_cue > 0:
                    label = "C"
                elif self.filtr in ["scope", "cuescope"] and self.in_xcope > 0:
                    label = "S"
                else:
                    label = "O"
                self.sentence.append((token, label))

    def startElement(self, name, attrs):
        if name == "cue":
            self.in_cue += 1
        elif name == "xcope":
            self.in_xcope += 1
        elif name == "sentence":
            self.in_cue = 0
            self.in_xcope = 0
            self.in_sentence = True
            self.sentence = []

    def endElement(self, name):
        if name == "cue":
            self.in_cue -= 1
        elif name == "xcope":
            self.in_xcope -= 1
        elif name == "sentence":
            self.in_sentence = False
            self.data.append(self.sentence)


def process_m2(m2_path, target_path, include_error_types):
    """
    Adapted from code by M. Rei
    """
    with open(m2_path, "r") as m2:
        with open(target_path, "w") as tsv:
            tsv_writer = csv.writer(tsv, delimiter="\t")
            while True:
                sentence = None
                errors = []
                for line in m2:
                    if line.startswith("S "):
                        sentence = line.strip().split()[1:]
                        errors = ["c" for i in range(len(sentence))]
                    elif line.startswith("A "):
                        line = line[2:]
                        line_parts = line.split("|||")
                        index_from = int(line_parts[0].split()[0])
                        index_to = int(line_parts[0].split()[1])
                        if index_from == -1 and index_to == -1:
                            continue
                        if index_from == index_to:
                            index_to = index_to + 1
                        error_type = line_parts[1]
                        for i in range(index_from, min(index_to, len(sentence))):
                            if include_error_types == True:
                                errors[i] = error_type
                            else:
                                errors[i] = "i"
                    elif sentence == None and len(line.strip()) == 0:
                        continue
                    elif sentence != None and len(line.strip()) == 0:
                        break
                    else:
                        raise ValueError("Unknown format")
                if sentence != None:
                    for i in range(len(sentence)):
                        tsv_writer.writerow([sentence[i], errors[i]])
                    tsv_writer.writerow([])

                else:
                    break


def process_conll10_xml(xml_dir, target_dir, mode, filtr):
    """
    Adapted from code by M. Rei
    """
    random.seed(123)

    if mode == "train":
        input_files = [
            os.path.join(xml_dir, "task2_train_bio_abstracts_rev2.xml"),
            os.path.join(xml_dir, "task2_train_bio_fullarticles_rev2.xml"),
        ]
        output_files = [
            os.path.join(target_dir, "conll_10_{}_train.tsv".format(filtr)),
            os.path.join(target_dir, "conll_10_{}_dev.tsv".format(filtr)),
        ]
    else:
        input_files = [os.path.join(xml_dir, "task2_eval_rev.xml")]
        output_files = [os.path.join(target_dir, "conll_10_{}_test.tsv".format(filtr))]

    data = []
    for path in input_files:
        data += XMLHandler(filtr).parse(path)

    if mode == "train":
        random.shuffle(data)
        dev_size = int(len(data) / 10.0)
        data_dev = data[:dev_size]
        data_train = data[dev_size:]

        with open(output_files[0], "w") as f:
            for sentence in data_train:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")

        with open(output_files[1], "w") as f:
            for sentence in data_dev:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")
    elif mode == "test":
        with open(output_files[0], "w") as f:
            for sentence in data:
                for token, label in sentence:
                    f.write(token + "\t" + label + "\n")
                f.write("\n")
    else:
        raise ValueError("Unknown")


def process_toxic_csv(csv_url, target_path):
    df = pd.read_csv(csv_url)

    with open(target_path, "w") as tsv:
        tsv_writer = csv.writer(tsv, delimiter="\t")
        for index, row in df.iterrows():
            text = row["text"]
            spans = ast.literal_eval(row["spans"])

            words = []
            toxics = []

            word = ""
            toxic = "f"

            for i in range(len(text)):
                if text[i] in [" ", "\n", "\t"]:
                    if len(word) != 0:
                        words.append(word)
                        toxics.append(toxic)

                    word = ""
                    toxic = "f"
                    continue

                elif text[i] in [".", ",", "!", "?", ":", ";", '"', "'"]:
                    if len(word) != 0:
                        words.append(word)
                        toxics.append(toxic)

                    word = ""
                    toxic = "f"

                word += text[i]
                if i in spans:
                    toxic = "t"

            for i in range(len(words)):
                tsv_writer.writerow([words[i], toxics[i]])
            tsv_writer.writerow([])


if __name__ == "__main__":
    dataset_names = ["fce", "conll_10", "toxic", "wi_locness"]

    parser = argparse.ArgumentParser(description="Dataset downloader and unzipper")
    parser.add_argument("-a", "--all", action="store_true", help="Process all datasets")
    parser.add_argument(
        "-d",
        "--data",
        action="store",
        nargs="+",
        default=[],
        help="Process specific datasets:{}".format(dataset_names),
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        default="../../data/processed",
        help="Target directory to process to",
    )
    parser.add_argument(
        "-s",
        "--source",
        action="store",
        default="../../data/raw",
        help="Target directory to process to",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        default=False,
        help="Removes any raw files after processing",
    )
    parser.add_argument(
        "-e",
        "--errors",
        action="store_true",
        default=False,
        help="Includes error types in tsv file",
    )

    args = parser.parse_args()

    download_all = args.all
    download_datasets = args.data
    target_path = args.target
    source_path = args.source
    cleanup = args.clean
    include_error_types = args.errors

    if download_all or "fce" in download_datasets:
        m2_dir = os.path.join(source_path, "fce_v2.1", "fce", "m2")

        if not os.path.exists(os.path.join(target_path, "fce_v2.1")):
            os.mkdir(os.path.join(target_path, "fce_v2.1"))

        process_m2(
            os.path.join(m2_dir, "fce.dev.gold.bea19.m2"),
            os.path.join(target_path, "fce_v2.1", "fce_dev.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "fce.test.gold.bea19.m2"),
            os.path.join(target_path, "fce_v2.1", "fce_test.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "fce.train.gold.bea19.m2"),
            os.path.join(target_path, "fce_v2.1", "fce_train.tsv"),
            include_error_types,
        )

        if cleanup:
            shutil.rmtree(os.path.join(source_path, "fce_v2.1"))

    if download_all or "conll_10" in download_datasets:
        xml_dir = os.path.join(source_path, "conll_10")

        if not os.path.exists(os.path.join(target_path, "conll_10")):
            os.mkdir(os.path.join(target_path, "conll_10"))

        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "train", "cuescope"
        )
        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "test", "cuescope"
        )

        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "train", "cue"
        )
        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "test", "cue"
        )

        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "train", "scope"
        )
        process_conll10_xml(
            xml_dir, os.path.join(target_path, "conll_10"), "test", "scope"
        )

        if cleanup:
            shutil.rmtree(os.path.join(source_path, "conll_10"))

    if download_all or "toxic" in download_datasets:
        csv_dir = os.path.join(source_path, "toxic")

        if not os.path.exists(os.path.join(target_path, "toxic")):
            os.mkdir(os.path.join(target_path, "toxic"))

        url = "https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data/"

        if not os.path.exists(os.path.join(target_path, "toxic")):
            os.mkdir(os.path.join(target_path, "toxic"))

        process_toxic_csv(
            url + "tsd_trial.csv", os.path.join(target_path, "toxic", "tsd_dev.tsv")
        )
        process_toxic_csv(
            url + "tsd_train.csv", os.path.join(target_path, "toxic", "tsd_train.tsv")
        )
        process_toxic_csv(
            url + "tsd_test.csv", os.path.join(target_path, "toxic", "tsd_test.tsv")
        )

    if download_all or "wi_locness" in download_datasets:
        m2_dir = os.path.join(source_path, "wi_locness_v2.1", "wi+locness", "m2")

        if not os.path.exists(os.path.join(target_path, "wi_locness_v2.1")):
            os.mkdir(os.path.join(target_path, "wi_locness_v2.1"))

        process_m2(
            os.path.join(m2_dir, "A.dev.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_A_dev.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "A.train.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_A_train.tsv"),
            include_error_types,
        )

        process_m2(
            os.path.join(m2_dir, "B.dev.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_B_dev.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "B.train.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_B_train.tsv"),
            include_error_types,
        )

        process_m2(
            os.path.join(m2_dir, "C.dev.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_C_dev.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "C.train.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_C_train.tsv"),
            include_error_types,
        )

        process_m2(
            os.path.join(m2_dir, "N.dev.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_N_dev.tsv"),
            include_error_types,
        )

        process_m2(
            os.path.join(m2_dir, "ABCN.dev.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_ABCN_dev.tsv"),
            include_error_types,
        )
        process_m2(
            os.path.join(m2_dir, "ABC.train.gold.bea19.m2"),
            os.path.join(target_path, "wi_locness_v2.1", "wi_locness_ABC_train.tsv"),
            include_error_types,
        )

        if cleanup:
            shutil.rmtree(os.path.join(source_path, "wi_locness_v2.1"))
