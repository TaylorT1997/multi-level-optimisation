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
    BertTokenizer,
    AdamW,
    BertConfig,
    DebertaTokenizer,
    DebertaForSequenceClassification,
    DebertaConfig,
    GPT2Tokenizer,
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
        print("Testing model: {}".format(args.model_name))
        print("*" * 30)
        print()
        print("Model: {}".format(args.model))
        print("Tokenizer: {}".format(args.tokenizer))
        print("Dataset: {}".format(args.dataset))
        print("Batch size: {}".format(args.batch_size))
        print()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if "bert-base" in args.model:
        if args.mlo_model:
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                sentence_loss_weight=args.sentence_loss_weight,
                token_loss_weight=args.token_loss_weight,
                regularizer_loss_weight=args.regularizer_loss_weight,
                token_supervision=args.token_supervision,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
            )
        else:
            model_config = BertConfig.from_pretrained(args.model, num_labels=1)
            model = BertForSequenceClassification(model_config)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    elif "deberta-base" in args.model:
        if args.mlo_model:
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                sentence_loss_weight=args.sentence_loss_weight,
                token_loss_weight=args.token_loss_weight,
                regularizer_loss_weight=args.regularizer_loss_weight,
                token_supervision=args.token_supervision,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
            )
        else:
            model_config = DebertaConfig.from_pretrained(args.model, num_labels=1)
            model = DebertaForSequenceClassification(model_config)
        tokenizer = DebertaTokenizer.from_pretrained(args.tokenizer)

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=1, bias=True), torch.nn.Sigmoid()
    )

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

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Evaluation loop
    model.eval()
    val_samples = 0
    val_total_loss = 0
    val_batches = 0

    val_seq_true_positives = 0
    val_seq_false_positives = 0
    val_seq_true_negatives = 0
    val_seq_false_negatives = 0

    val_token_true_positives = 0
    val_token_false_positives = 0
    val_token_true_negatives = 0
    val_token_false_negatives = 0

    if args.use_wandb:
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
            # Zero any accumulated gradients

            # Batch decode and encode to get correct padding for the batch
            decoded = tokenizer.batch_decode(sequences)
            encoded_sequences = tokenizer(
                decoded, padding=True, truncation=True, return_tensors="pt",
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
            labels = torch.tensor(labels, dtype=torch.float, device=device)
            token_labels = torch.tensor(token_labels, dtype=torch.float, device=device)

            # If using mlo model pass inputs and token labels through mlo model
            if args.mlo_model:
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=token_labels
                )
                seq_logits = outputs["sequence_logits"]
                token_logits = outputs["token_logits"]

            # Otherwise pass inputs and sequence labels through basic pretrained model
            else:
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=labels
                )
                seq_logits = outputs.logits

            # Calculate token prediction metrics
            if args.mlo_model:
                token_preds = token_logits > 0.5
                token_actuals = token_labels

                masked_zeros = torch.where(
                    token_actuals != -1, token_actuals, torch.zeros_like(token_actuals),
                )
                masked_ones = torch.where(
                    token_actuals != -1, token_actuals, torch.ones_like(token_actuals),
                )

                token_true_positives = torch.sum(
                    torch.logical_and(token_preds == 1, masked_zeros == 1)
                ).item()
                token_false_positives = torch.sum(
                    torch.logical_and(token_preds == 1, masked_ones == 0)
                ).item()
                token_true_negatives = torch.sum(
                    torch.logical_and(token_preds == 0, masked_ones == 0)
                ).item()
                token_false_negatives = torch.sum(
                    torch.logical_and(token_preds == 0, masked_zeros == 1)
                ).item()

                val_token_true_positives += token_true_positives
                val_token_false_positives += token_false_positives
                val_token_true_negatives += token_true_negatives
                val_token_false_negatives += token_false_negatives

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

            val_seq_true_positives += seq_true_positives
            val_seq_false_positives += seq_false_positives
            val_seq_true_negatives += seq_true_negatives
            val_seq_false_negatives += seq_false_negatives

            val_samples += len(seq_preds)
            val_total_loss += loss.item()
            val_batches += 1

    # Calculate validation metrics
    seq_val_accuracy = (val_seq_true_positives + val_seq_true_negatives) / val_samples
    seq_val_precision = val_seq_true_positives / (
        val_seq_true_positives + val_seq_false_positives + 1e-5
    )
    seq_val_recall = val_seq_true_positives / (
        val_seq_true_positives + val_seq_false_negatives + 1e-5
    )
    seq_val_f1 = (2 * seq_val_precision * seq_val_recall) / (
        seq_val_precision + seq_val_recall + 1e-5
    )
    seq_val_f05 = (1.25 * seq_val_precision * seq_val_recall) / (
        0.25 * seq_val_precision + seq_val_recall + 1e-5
    )

    token_val_accuracy = (val_token_true_positives + val_token_true_negatives) / (
        val_token_true_positives
        + val_token_false_positives
        + val_token_true_negatives
        + val_token_false_negatives
    )
    token_val_precision = val_token_true_positives / (
        val_token_true_positives + val_token_false_positives + 1e-5
    )
    token_val_recall = val_token_true_positives / (
        val_token_true_positives + val_token_false_negatives + 1e-5
    )
    token_val_f1 = (2 * token_val_precision * token_val_recall) / (
        token_val_precision + token_val_recall + 1e-5
    )
    token_val_f05 = (1.25 * token_val_precision * token_val_recall) / (
        0.25 * token_val_precision + token_val_recall + 1e-5
    )

    val_av_loss = val_total_loss / val_batches

    if not args.silent:
        print()
        print("-" * 30)
        print("Epoch {} / {}".format(epoch, args.epochs))
        print("-" * 30)
        print()
        print("Training loss: {:.4f}".format(train_av_loss))
        print("Validation loss: {:.4f}".format(val_av_loss))
        print()
        print("Validation sequence accuracy: {:.4f}".format(seq_val_accuracy))
        print("Validation sequence precision: {:.4f}".format(seq_val_precision))
        print("Validation sequence recall: {:.4f}".format(seq_val_recall))
        print("Validation sequence f1: {:.4f}".format(seq_val_f1))
        print("Validation sequence f0.5: {:.4f}".format(seq_val_f05))
        print()
        if args.mlo_model:
            print("Validation token accuracy: {:.4f}".format(token_val_accuracy))
            print("Validation token precision: {:.4f}".format(token_val_precision))
            print("Validation token recall: {:.4f}".format(token_val_recall))
            print("Validation token f1: {:.4f}".format(token_val_f1))
            print("Validation token f0.5: {:.4f}".format(token_val_f05))
            print()


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        default_config_files=[
            "/home/taylort/Projects/multi-level-optimisation/config.txt"
        ]
    )

    parser.add(
        "-c",
        "--config",
        is_config_file=True,
        default="/home/taylort/Projects/multi-level-optimisation/config.txt",
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
        help="Pretrained model to use",
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
        "--model_name",
        action="store",
        type=str,
        default=None,
        help="Name of model being evaluated",
    )

    args = parser.parse_args()
    test(args)
