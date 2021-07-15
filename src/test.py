import sys
import argparse
import time
import datetime
import os
import configargparse
import math
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    AutoConfig,
)

from data_loading.datasets import BinaryTokenTSVDataset
from models.model import TokenModel
from models.seq_class_model import SeqClassModel

from sklearn.metrics import average_precision_score

import wandb


def collate_fn(batch):
    sequences, labels, token_labels = [], [], []
    for sequence, label, token_label in batch:
        sequences.append(sequence)
        labels.append(label)
        token_labels.append(token_label)
    return sequences, labels, token_labels


def test(args):
    torch.manual_seed(args.seed)

    if not args.silent:
        print("*" * 43)
        print("Testing model: {}".format(args.model_path))
        print("*" * 43)
        print()
        print("Model: {}".format(args.model))
        print("Model architecture: {}".format(args.model_architecture))
        print("Tokenizer: {}".format(args.tokenizer))
        print("Dataset: {}".format(args.dataset))
        print("Batch size: {}".format(args.batch_size))
        print()
        print("Maximum sequence length: {}".format(args.max_sequence_length))
        print("Soft attention beta: {}".format(args.soft_attention_beta))
        print("Subword method: {}".format(args.subword_method))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set the tokenizer
    if "bert-base" in args.model:
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    elif "deberta-base" in args.model:
        tokenizer = DebertaTokenizer.from_pretrained(args.tokenizer)
    elif "roberta-base" in args.model:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.tokenizer, add_prefix_space=True
        )

    # Define test dataset
    if "wi_locness" in args.dataset:
        test_dataset = BinaryTokenTSVDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            root_dir=args.root,
            token_label_mode="first",
            mode="dev",
            wi_locness_type="ABCN",
            include_special_tokens=False,
            use_lowercase=args.use_lowercase,
            max_sequence_length=args.max_sequence_length,
        )

    else:
        test_dataset = BinaryTokenTSVDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            root_dir=args.root,
            token_label_mode="first",
            mode="test",
            include_special_tokens=False,
            use_lowercase=args.use_lowercase,
            max_sequence_length=args.max_sequence_length,
        )

    negative_label = test_dataset.negative_label
    positive_label = test_dataset.positive_label

    print()
    print(f"Testing on dataset of length {len(test_dataset)}")
    print()

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    zero_shot_config_dict = {
        "experiment_name": "final_soft_attention",
        "dataset": args.dataset,
        "model_name": args.model,
        "max_seq_length": args.max_sequence_length,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 4,
        "seed": args.seed,
        "lowercase": args.use_lowercase,
        "gradient_accumulation_steps": 1,
        "save_steps": 500,
        "logging_steps": 500,
        "do_mask_words": False,
        "mask_prob": 0.0,
        "hid_to_attn_dropout": 0.10,
        "attention_evidence_size": 100,
        "final_hidden_layer_size": 300,
        "initializer_name": "glorot",
        "attention_activation": "soft",
        "soft_attention": True,
        "soft_attention_alpha": 0.1,
        "soft_attention_gamma": 0.1,
        "soft_attention_beta": 0.0,
        "square_attention": True,
        "freeze_bert_layers_up_to": 0,
        "zero_n": 0,
        "zero_delta": 0.0,
    }

    # Define model back bone and architecture
    if "bert-base" in args.model:
        if args.model_architecture == "joint":
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                sentence_loss_weight=args.sentence_loss_weight,
                token_loss_weight=args.token_loss_weight,
                regularizer_loss_weight=args.regularizer_loss_weight,
                token_supervision=args.token_supervision,
                sequence_supervision=args.sequence_supervision,
                regularization_losses=args.regularization_losses,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
                subword_method=args.subword_method,
                device=device,
                debug=args.debug,
            )
        elif args.model_architecture == "zero_shot":
            labels = [negative_label, positive_label]
            label_map = {i: label for i, label in enumerate(labels)}

            config = AutoConfig.from_pretrained(
                args.model,
                id2label=label_map,
                label2id={label: i for i, label in enumerate(labels)},
                output_hidden_states=True,
                output_attentions=True,
            )
            model = SeqClassModel(
                params_dict=zero_shot_config_dict, model_config=config
            )
        elif args.model_architecture == "base":
            model_config = BertConfig.from_pretrained(args.model, num_labels=2)
            model = BertForSequenceClassification(model_config)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=768, out_features=1, bias=True),
                torch.nn.Sigmoid(),
            )
    elif "deberta-base" in args.model:
        if args.model_architecture == "joint":
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                sentence_loss_weight=args.sentence_loss_weight,
                token_loss_weight=args.token_loss_weight,
                regularizer_loss_weight=args.regularizer_loss_weight,
                token_supervision=args.token_supervision,
                sequence_supervision=args.sequence_supervision,
                regularization_losses=args.regularization_losses,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
                subword_method=args.subword_method,
                device=device,
                debug=args.debug,
            )
        elif args.model_architecture == "zero_shot":
            labels = [negative_label, positive_label]
            label_map = {i: label for i, label in enumerate(labels)}

            config = AutoConfig.from_pretrained(
                args.model,
                id2label=label_map,
                label2id={label: i for i, label in enumerate(labels)},
                output_hidden_states=True,
                output_attentions=True,
            )
            model = SeqClassModel(
                params_dict=zero_shot_config_dict, model_config=config
            )
        elif args.model_architecture == "base":
            model_config = DebertaConfig.from_pretrained(args.model, num_labels=2)
            model = DebertaForSequenceClassification(model_config)
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=768, out_features=1, bias=True),
                torch.nn.Sigmoid(),
            )

    elif "roberta-base" in args.model:
        if args.model_architecture == "joint":
            model = TokenModel(
                pretrained_model=args.model,
                soft_attention_beta=args.soft_attention_beta,
                sentence_loss_weight=args.sentence_loss_weight,
                token_loss_weight=args.token_loss_weight,
                regularizer_loss_weight=args.regularizer_loss_weight,
                token_supervision=args.token_supervision,
                sequence_supervision=args.sequence_supervision,
                regularization_losses=args.regularization_losses,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
                subword_method=args.subword_method,
                device=device,
                debug=args.debug,
            )
        elif args.model_architecture == "zero_shot":
            labels = [negative_label, positive_label]
            label_map = {i: label for i, label in enumerate(labels)}

            config = AutoConfig.from_pretrained(
                args.model,
                id2label=label_map,
                label2id={label: i for i, label in enumerate(labels)},
                output_hidden_states=True,
                output_attentions=True,
            )
            model = SeqClassModel(
                params_dict=zero_shot_config_dict, model_config=config
            )
        elif args.model_architecture == "base":
            model_config = RobertaConfig.from_pretrained(args.model, num_labels=2)
            model = RobertaForSequenceClassification(model_config)

    # Load model from path and put on device
    model.load_state_dict(
        torch.load(os.path.join(args.root, "models", args.model_path, "model.pt"))
    )
    model.to(device)

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

    test_token_total_ap = 0
    num_map_scores = 0

    preds = torch.tensor([], device=device)
    actuals = torch.tensor([], device=device)

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
                max_length=args.max_sequence_length,
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

            num_labels = len(labels)

            # If using mlo model pass inputs and token labels through mlo model
            if args.model_architecture == "joint":
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

            elif args.model_architecture == "zero_shot":
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=labels.long(), token_labels=token_labels
                )
                loss, logits, token_logits = outputs
                seq_logits = torch.argmax(logits, dim=1)

            elif args.model_architecture == "base":
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=labels.long(),
                )
                loss = outputs.loss
                seq_logits = torch.argmax(outputs.logits, dim=1)

            # Calculate token prediction metrics
            if (
                args.model_architecture == "joint"
                or args.model_architecture == "zero_shot"
            ):
                # Calculate TP, FP, TN, FN
                token_preds = (token_logits > 0.5).long()

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

                # Calculate mAP
                token_labels_np = token_labels.detach().cpu().numpy()
                token_logits_np = token_logits.detach().cpu().numpy()

                for i in range(len(token_labels_np)):
                    if np.max(token_labels_np[i]) == 1:
                        ap = average_precision_score(
                            token_labels_np[i][
                                (token_labels_np[i] == 1) | (token_labels_np[i] == 0)
                            ],
                            token_logits_np[i][
                                (token_labels_np[i] == 1) | (token_labels_np[i] == 0)
                            ],
                        )

                        test_token_total_ap += ap
                        num_map_scores += 1

            # Calculate sequence prediction metrics
            seq_preds = (seq_logits.view(-1) > 0.5).long()
            seq_actuals = labels

            preds = torch.cat((preds, seq_preds), 0)
            actuals = torch.cat((actuals, seq_actuals), 0)

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

    # Calculate test metrics
    seq_test_accuracy = (test_seq_true_positives + test_seq_true_negatives) / (
        test_seq_true_positives
        + test_seq_true_negatives
        + test_seq_false_positives
        + test_seq_false_negatives
        + 1e-99
    )
    seq_test_precision = test_seq_true_positives / (
        test_seq_true_positives + test_seq_false_positives + 1e-99
    )
    seq_test_recall = test_seq_true_positives / (
        test_seq_true_positives + test_seq_false_negatives + 1e-99
    )
    seq_test_f1 = (2 * seq_test_precision * seq_test_recall) / (
        seq_test_precision + seq_test_recall + 1e-99
    )
    seq_test_f05 = (1.25 * seq_test_precision * seq_test_recall) / (
        0.25 * seq_test_precision + seq_test_recall + 1e-99
    )

    token_test_accuracy = (test_token_true_positives + test_token_true_negatives) / (
        test_token_true_positives
        + test_token_false_positives
        + test_token_true_negatives
        + test_token_false_negatives
        + 1e-99
    )
    token_test_precision = test_token_true_positives / (
        test_token_true_positives + test_token_false_positives + 1e-99
    )
    token_test_recall = test_token_true_positives / (
        test_token_true_positives + test_token_false_negatives + 1e-99
    )
    token_test_f1 = (2 * token_test_precision * token_test_recall) / (
        token_test_precision + token_test_recall + 1e-99
    )
    token_test_f05 = (1.25 * token_test_precision * token_test_recall) / (
        0.25 * token_test_precision + token_test_recall + 1e-99
    )

    token_test_map = test_token_total_ap / num_map_scores

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
                "token_map": token_test_map,
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
        if args.model_architecture == "joint" or args.model_architecture == "zero_shot":
            print("Test token accuracy: {:.4f}".format(token_test_accuracy))
            print("Test token precision: {:.4f}".format(token_test_precision))
            print("Test token recall: {:.4f}".format(token_test_recall))
            print("Test token f1: {:.4f}".format(token_test_f1))
            print("Test token f0.5: {:.4f}".format(token_test_f05))
            print("Test token mAP: {:.4f}".format(token_test_map))
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
        "--model_architecture",
        action="store",
        type=str,
        default="joint",
        help="Model architecture to use (default: joint)",
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

    parser.add(
        "--max_sequence_length",
        action="store",
        type=int,
        default=512,
        help="Maximum sequence length to input to model (default: 512)",
    )

    parser.add(
        "--use_lowercase",
        action="store_true",
        default=False,
        help="Use lowercase as input (default: False)",
    )

    parser.add(
        "--seed", action="store", type=int, default=666, help="Random seed for model",
    )

    args = parser.parse_args()
    test(args)
