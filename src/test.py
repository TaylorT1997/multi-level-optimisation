import sys
import argparse
import time
import datetime
import os
import configargparse
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    RobertaConfig,
    AdamW,
    BertConfig,
    DebertaTokenizer,
    DebertaForSequenceClassification,
    DebertaConfig,
    GPT2TokenizerFast,
)

from data_loading.datasets import BinaryTokenTSVDataset
from models.model import TokenModel

import wandb


def collate_fn(batch):
    sequences, labels, token_labels = [], [], []
    for sequence, label, token_label in batch:
        sequences.append(sequence)
        labels.append(label)
        token_labels.append(token_label)
    return sequences, labels, token_labels


def test(args):
    torch.manual_seed(666)

    if not args.silent:
        print("*" * 30)
        print("Testing model: {}".format(args.model_path.split("/")[-2]))
        print("*" * 30)
        print()
        print("Model: {}".format(args.model))
        print("Tokenizer: {}".format(args.tokenizer))
        print("Dataset: {}".format(args.dataset))
        print("Batch size: {}".format(args.batch_size))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define model
    if "bert-base" in args.model:
        if args.mlo_model:
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                subword_method=args.subword_method,
                device=device,
            )
        else:
            model_config = BertConfig.from_pretrained(args.model, num_labels=1)
            model = BertForSequenceClassification(model_config)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=768, out_features=1, bias=True),
                torch.nn.Sigmoid(),
            )
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    elif "deberta-base" in args.model:
        if args.mlo_model:
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                subword_method=args.subword_method,
                device=device,
            )
        else:
            model_config = DebertaConfig.from_pretrained(args.model, num_labels=1)
            model = DebertaForSequenceClassification(model_config)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=768, out_features=1, bias=True),
                torch.nn.Sigmoid(),
            )
        tokenizer = DebertaTokenizer.from_pretrained(args.tokenizer)
    elif "roberta-base" in args.model:
        if args.mlo_model:
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                subword_method=args.subword_method,
                device=device,
            )
        else:
            model_config = RobertaConfig.from_pretrained(args.model, num_labels=1)
            model = RobertaForSequenceClassification(model_config)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=768, out_features=1, bias=True),
                torch.nn.Sigmoid(),
            )
        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.tokenizer, add_prefix_space=True
        )

    # Load model from path and put on device
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Define test dataset
    if "wi_locness" in args.dataset:
        test_dataset = BinaryTokenTSVDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            root_dir=args.root,
            mode="dev",
            include_special_tokens=False,
        )

    else:
        test_dataset = BinaryTokenTSVDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            root_dir=args.root,
            mode="test",
            include_special_tokens=False,
        )

    print()
    print(f"Testing on dataset of length {len(test_dataset)}")
    print()

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Evaluation loop
    model.eval()

    test_seq_true_positives = 0
    test_seq_false_positives = 0
    test_seq_true_negatives = 0
    test_seq_false_negatives = 0

    test_token_true_positives = 0
    test_token_false_positives = 0
    test_token_true_negatives = 0
    test_token_false_negatives = 0

    # Log with wandb
    if args.use_wandb:
        wandb.init(project="multi-level-optimisation", entity="taylort1997")
        wandb.config.update(args)
        wandb.watch(model)
        if args.mlo_model:
            table = wandb.Table(
                columns=[
                    "Input Text",
                    "Predicted Token Labels",
                    "True Token Labels",
                    "Predicted Label",
                    "True Label",
                ]
            )
        else:
            table = wandb.Table(columns=["Input Text", "Predicted Label", "True Label"])

    with torch.no_grad():
        for idx, (sequences, labels, token_labels) in enumerate(test_loader):

            # Batch encode to get correct padding for the batch
            encoded_sequences = tokenizer(
                sequences,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Pad token labels
            max_length = 0
            for seq in encoded_sequences["input_ids"]:
                if len(seq) > max_length:
                    max_length = len(seq)

            for i in range(len(token_labels)):
                token_labels[i].extend([-1] * (max_length - len(token_labels[i])))
                token_labels[i] = token_labels[i][:max_length]

            # Gather input_ids, attention masks, sequence labels and token labels from batch
            input_ids = encoded_sequences["input_ids"].to(device)
            attention_masks = encoded_sequences["attention_mask"].to(device)
            offset_mapping = encoded_sequences["offset_mapping"].to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)
            token_labels = torch.tensor(token_labels, dtype=torch.float, device=device)

            # If using mlo model pass inputs and token labels through mlo model
            if args.mlo_model:
                outputs = model(
                    input_ids,
                    attention_mask=attention_masks,
                    offset_mapping=offset_mapping,
                    labels=token_labels,
                )
                loss = outputs["loss"]
                sentence_loss = outputs["sentence_loss"]
                token_loss = outputs["token_loss"]
                regularizer_loss_a = outputs["regularizer_loss_a"]
                regularizer_loss_b = outputs["regularizer_loss_b"]
                seq_logits = outputs["sequence_logits"]
                token_logits = outputs["token_logits"]

            # Otherwise pass inputs and sequence labels through basic pretrained model
            else:
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=labels
                )
                loss = outputs.loss
                seq_logits = outputs.logits

            # print(seq_logits)
            # print(labels)

            # Calculate token prediction metrics
            if args.mlo_model:
                token_preds = token_logits > 0.5

                token_true_positives = torch.sum(
                    torch.logical_and(token_preds == 1, token_labels == 1)
                ).item()
                token_false_positives = torch.sum(
                    torch.logical_and(token_preds == 1, token_labels == 0)
                ).item()
                token_true_negatives = torch.sum(
                    torch.logical_and(token_preds == 0, token_labels == 0)
                ).item()
                token_false_negatives = torch.sum(
                    torch.logical_and(token_preds == 0, token_labels == 1)
                ).item()

                test_token_true_positives += token_true_positives
                test_token_false_positives += token_false_positives
                test_token_true_negatives += token_true_negatives
                test_token_false_negatives += token_false_negatives

            # Calculate sequence prediction metrics
            seq_preds = seq_logits.view(-1) > 0.5
            seq_actuals = labels

            seq_true_positives = torch.sum(
                torch.logical_and(seq_preds == 1, seq_actuals == 1)
            ).item()
            seq_false_positives = torch.sum(
                torch.logical_and(seq_preds == 1, seq_actuals == 0)
            ).item()
            seq_true_negatives = torch.sum(
                torch.logical_and(seq_preds == 0, seq_actuals == 0)
            ).item()
            seq_false_negatives = torch.sum(
                torch.logical_and(seq_preds == 0, seq_actuals == 1)
            ).item()

            test_seq_true_positives += seq_true_positives
            test_seq_false_positives += seq_false_positives
            test_seq_true_negatives += seq_true_negatives
            test_seq_false_negatives += seq_false_negatives

            if args.use_wandb:
                for i in range(len(labels)):
                    if "deberta" in args.tokenizer:
                        # Need to use gpt2 tokenizer due to bug with deberta tokenizer
                        input_text = [
                            tokenizer.gpt2_tokenizer.decode(
                                [tokenizer.gpt2_tokenizer.sym(id)]
                            )
                            if tokenizer.gpt2_tokenizer.sym(id)
                            not in tokenizer.all_special_tokens
                            else tokenizer.gpt2_tokenizer.sym(id)
                            for id in input_ids[i]
                        ]
                        input_text = list(
                            filter(
                                lambda x: x != "[CLS]"
                                and x != "[SEP]"
                                and x != "[PAD]",
                                input_text,
                            )
                        )
                        input_text = " ".join(input_text)
                    else:
                        input_text = tokenizer.decode(
                            input_ids[i],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    true_label = seq_actuals[i].item()
                    pred_label = seq_preds[i].item()

                    if args.mlo_model:
                        true_token_labels = token_labels[i][token_labels[i] != -1]
                        pred_token_labels = token_preds[i][token_labels[i] != -1]
                        table.add_data(
                            input_text,
                            str(pred_label),
                            str(true_label),
                            str(pred_token_labels),
                            str(true_token_labels),
                        )
                    else:
                        table.add_data(input_text, str(pred_label), str(true_label))
                wandb.log({f"test samples: {args.dataset}": table})

    # Calculate test metrics
    seq_test_accuracy = (test_seq_true_positives + test_seq_true_negatives) / (
        test_seq_true_positives
        + test_seq_true_negatives
        + test_seq_true_negatives
        + test_seq_false_negatives
        + 1e-5
    )
    seq_test_precision = test_seq_true_positives / (
        test_seq_true_positives + test_seq_false_positives + 1e-5
    )
    seq_test_recall = test_seq_true_positives / (
        test_seq_true_positives + test_seq_false_negatives + 1e-5
    )
    seq_test_f1 = (2 * seq_test_precision * seq_test_recall) / (
        seq_test_precision + seq_test_recall + 1e-5
    )
    seq_test_f05 = (1.25 * seq_test_precision * seq_test_recall) / (
        0.25 * seq_test_precision + seq_test_recall + 1e-5
    )

    token_test_accuracy = (test_token_true_positives + test_token_true_negatives) / (
        test_token_true_positives
        + test_token_false_positives
        + test_token_true_negatives
        + test_token_false_negatives
        + 1e-5
    )
    token_test_precision = test_token_true_positives / (
        test_token_true_positives + test_token_false_positives + 1e-5
    )
    token_test_recall = test_token_true_positives / (
        test_token_true_positives + test_token_false_negatives + 1e-5
    )
    token_test_f1 = (2 * token_test_precision * token_test_recall) / (
        token_test_precision + token_test_recall + 1e-5
    )
    token_test_f05 = (1.25 * token_test_precision * token_test_recall) / (
        0.25 * token_test_precision + token_test_recall + 1e-5
    )

    if args.use_wandb:
        wandb.log(
            {
                "seq_accuracy_test": seq_test_accuracy,
                "seq_precision_test": seq_test_precision,
                "seq_recall_test": seq_test_recall,
                "seq_f1_test": seq_test_f1,
                "seq_f0.5_test": seq_test_f05,
                "token_accuracy_test": token_test_accuracy,
                "token_precision_test": token_test_precision,
                "token_recall_test": token_test_recall,
                "token_f1_test": token_test_f1,
                "token_f0.5_test": token_test_f05,
            }
        )

    if not args.silent:
        print()
        print("Test sequence accuracy: {:.4f}".format(seq_test_accuracy))
        print("Test sequence precision: {:.4f}".format(seq_test_precision))
        print("Test sequence recall: {:.4f}".format(seq_test_recall))
        print("Test sequence f1: {:.4f}".format(seq_test_f1))
        print("Test sequence f0.5: {:.4f}".format(seq_test_f05))
        print()
        if args.mlo_model:
            print("Test token accuracy: {:.4f}".format(token_test_accuracy))
            print("Test token precision: {:.4f}".format(token_test_precision))
            print("Test token recall: {:.4f}".format(token_test_recall))
            print("Test token f1: {:.4f}".format(token_test_f1))
            print("Test token f0.5: {:.4f}".format(token_test_f05))
            print()


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        default_config_files=[
            "/home/taylort/Projects/multi-level-optimisation/config_test.txt"
        ]
    )

    parser.add(
        "-c",
        "--config",
        is_config_file=True,
        default="/home/taylort/Projects/multi-level-optimisation/config_test.txt",
        help="Config file path",
    )

    parser.add(
        "-r",
        "--root",
        action="store",
        type=str,
        default="/home/tom/Projects/multi-level-optimisation/",
        help="Root directory of project",
    )

    parser.add(
        "--model",
        action="store",
        type=str,
        default="bert-base-cased",
        help="Pretrained model to use (default: bert-base-cased",
    )

    parser.add(
        "--mlo_model",
        action="store_true",
        default=False,
        help="Use multi-level optimisation model (default: False)",
    )

    parser.add(
        "--tokenizer",
        action="store",
        type=str,
        default="bert-base-cased",
        help="Pretrained tokenizer to use",
    )

    parser.add(
        "--dataset",
        action="store",
        type=str,
        default="conll_10",
        help="Dataset to train on",
    )

    parser.add(
        "--batch_size",
        action="store",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )

    parser.add(
        "-p",
        "--model_path",
        action="store",
        type=str,
        default=None,
        help="Path of pretrained model to evaluate",
    )

    parser.add(
        "--soft_attention_beta",
        action="store",
        type=float,
        default=1,
        help="Soft attention beta value (default: 1)",
    )

    parser.add(
        "--subword_method",
        action="store",
        type=str,
        default="first",
        help="Method for dealing with subwords (default: first)",
    )

    parser.add(
        "-w",
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use wandb to track run (default: False)",
    )

    parser.add(
        "-s",
        "--silent",
        action="store_true",
        default=False,
        help="Silence console prints (default: False)",
    )

    args = parser.parse_args()
    test(args)
