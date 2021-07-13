import sys
import argparse
import time
import datetime
import os
import configargparse
import math

import torch
from torch._C import device
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
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
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_scheduler,
)

from data_loading.datasets import BinaryTokenTSVDataset
from models.model import TokenModel

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    sequences, labels, token_labels = [], [], []
    for sequence, label, token_label in batch:
        sequences.append(sequence)
        labels.append(label)
        token_labels.append(token_label)
    return sequences, labels, token_labels


def train(args):
    torch.manual_seed(args.seed)

    if not args.silent:
        if args.debug:
            print("!" * 30)
            print("!" * 5 + " DEBUG MODE ENABLED " + "!" * 5)
            print("!" * 30)
            print()
        print("*" * 30)
        print(
            "Training: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        print("*" * 30)
        print()
        print("Model: {}".format(args.model))
        print("Multi-level optimisation: {}".format(args.mlo_model))
        print("Tokenizer: {}".format(args.tokenizer))
        print("Dataset: {}".format(args.dataset))
        print()
        print("Batch size: {}".format(args.batch_size))
        print("Epochs: {}".format(args.epochs))
        print("Early stopping patience: {}".format(args.early_stopping_patience))
        print("Early stopping objective: {}".format(args.early_stopping_objective))
        print()
        print("Learning rate: {}".format(args.learning_rate))
        print("Learning rate weight decay: {}".format(args.lr_weight_decay))
        print("Learning rate epsilon: {}".format(args.lr_epsilon))
        print()
        print("Optimizer: {}".format(args.lr_optimizer))
        print("Scheduler: {}".format(args.lr_scheduler))
        print("Scheduler step size: {}".format(args.lr_scheduler_step))
        print("Scheduler gamma: {}".format(args.lr_scheduler_gamma))
        print("Scheduler warmup ratio: {}".format(args.lr_scheduler_warmup_ratio))
        print()
        print("Maximum sequence length: {}".format(args.max_sequence_length))
        print("Soft attention beta: {}".format(args.soft_attention_beta))
        print("Sentence loss weight: {}".format(args.sentence_loss_weight))
        print("Token loss weight: {}".format(args.token_loss_weight))
        print("Regularizer loss weight: {}".format(args.regularizer_loss_weight))
        print("Token supervision: {}".format(args.token_supervision))
        print("Subword method: {}".format(args.subword_method))
        print("Use lowercase: {}".format(args.use_lowercase))
        print(
            "Normalise supervised losses: {}".format(args.normalise_supervised_losses)
        )
        print(
            "Normalise regularization losses: {}".format(
                args.normalise_regularization_losses
            )
        )

    if args.save_model:
        model_dir = os.path.join(
            args.root,
            "models",
            "{}_{}".format(
                args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            ),
        )
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

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
                sequence_supervision=args.sequence_supervision,
                regularization_losses=args.regularization_losses,
                normalise_supervised_losses=args.normalise_supervised_losses,
                normalise_regularization_losses=args.normalise_regularization_losses,
                subword_method=args.subword_method,
                device=device,
                debug=args.debug,
            )
        else:
            model_config = BertConfig.from_pretrained(args.model, num_labels=2)
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
        else:
            model_config = DebertaConfig.from_pretrained(args.model, num_labels=2)
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
        else:
            # model_config = RobertaConfig.from_pretrained(args.model, num_labels=2)
            # model = RobertaForSequenceClassification(model_config)

            from models.seq_class_model import SeqClassModel
            from transformers import AutoConfig

            config_dict = {
                "experiment_name": "final_soft_attention",
                "dataset": "conll10",
                "model_name": "roberta-base",
                "max_seq_length": 512,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 32,
                "num_train_epochs": 20,
                "warmup_ratio": 0.1,
                "learning_rate": 2e-5,
                "weight_decay": 0.1,
                "seed": 15,
                "adam_epsilon": 1e-7,
                "test_label_dummy": "test",
                "make_all_labels_equal_max": True,
                "is_seq_class": True,
                "lowercase": True,
                "gradient_accumulation_steps": 1,
                "save_steps": 500,
                "logging_steps": 500,
                "output_dir": "models/{experiment_name}/{model_name}/{dataset_name}/{datetime}/",
                "do_mask_words": False,
                "mask_prob": 0.0,
                "hid_to_attn_dropout": 0.10,
                "attention_evidence_size": 100,
                "final_hidden_layer_size": 300,
                "initializer_name": "glorot",
                "attention_activation": "soft",
                "soft_attention": True,
                "soft_attention_gamma": 0.1,
                "soft_attention_alpha": 0.1,
                "square_attention": True,
                "freeze_bert_layers_up_to": 0,
                "zero_n": 0,
                "zero_delta": 0.0,
            }

            labels = ["O", "C"]
            idx_pos = min([i for i, val in enumerate(labels) if val == "C"])
            label_map = {i: label for i, label in enumerate(labels)}

            config = AutoConfig.from_pretrained(
                config_dict["model_name"],
                id2label=label_map,
                label2id={label: i for i, label in enumerate(labels)},
                output_hidden_states=True,
                output_attentions=True,
            )
            model = SeqClassModel(params_dict=config_dict, model_config=config)


        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.tokenizer, add_prefix_space=True
        )

    model.to(device)

    # Define training and validation datasets
    train_dataset = BinaryTokenTSVDataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        root_dir=args.root,
        mode="train",
        token_label_mode="first",
        wi_locness_type="ABC",
        include_special_tokens=False,
        use_lowercase=args.use_lowercase,
        max_sequence_length=args.max_sequence_length,
    )

    if "wi_locness" in args.dataset:
        dev_indices_file = open("../wi_locness_dev_indices_ABC.txt")
        dev_indices = [int(i) for i in dev_indices_file.read().split(",")]
        all_indices = range(len(train_dataset))
        train_indices = list(set(all_indices) - set(dev_indices))
        val_dataset = Subset(train_dataset, dev_indices)
        train_dataset = Subset(train_dataset, train_indices)

    else:
        val_dataset = BinaryTokenTSVDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            root_dir=args.root,
            mode="dev",
            token_label_mode="first",
            include_special_tokens=False,
            use_lowercase=args.use_lowercase,
            max_sequence_length=args.max_sequence_length,
        )

    print()
    print(f"Training on dataset of length {len(train_dataset)}")
    print(f"Validating on dataset of length {len(val_dataset)}")
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Optimizer and scheduler
    if args.lr_optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            eps=args.lr_epsilon,
            weight_decay=args.lr_weight_decay,
        )
    elif args.lr_optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            eps=args.lr_epsilon,
            weight_decay=args.lr_weight_decay,
        )
    elif args.lr_optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.lr_momentum,
            weight_decay=args.lr_weight_decay,
        )

    if args.lr_scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma
        )
    elif args.lr_scheduler == "warmup_linear":
        # max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        train_steps = args.epochs * (len(train_dataset) // args.batch_size)
        warmup_steps = args.lr_scheduler_warmup_ratio * train_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps
        )

    # Record steps
    train_step = 0
    val_step = 0

    # Record best metrics
    best_epoch = 0
    best_val_loss = 1e5
    best_seq_val_f1 = -1
    best_token_val_f1 = -1
    best_seq_val_f05 = -1
    best_token_val_f05 = -1

    # Early stopping
    no_improvement_num = 0

    # Log with wandb
    if args.use_wandb:
        wandb.init(project="multi-level-optimisation", entity="taylort1997")
        wandb.config.update(args)
        wandb.watch(model)

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training loop
        epoch_start = time.time()
        model.train()

        train_total_loss = 0
        train_batches = 0

        train_seq_true_positives = 0
        train_seq_false_positives = 0
        train_seq_true_negatives = 0
        train_seq_false_negatives = 0

        train_token_true_positives = 0
        train_token_false_positives = 0
        train_token_true_negatives = 0
        train_token_false_negatives = 0

        for idx, (sequences, labels, token_labels) in enumerate(train_loader):
            # Zero any accumulated gradients
            optimizer.zero_grad()

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
                # outputs = model(
                #     input_ids, attention_mask=attention_masks, labels=labels.long(),
                # )
                # loss = outputs.loss
                # seq_logits = torch.argmax(outputs.logits, dim=1)

                outputs = model(input_ids, attention_mask=attention_masks, labels=labels.long(),)
                loss, logits, _ = outputs
                seq_logits = torch.argmax(logits, dim=1)


            # Backpropagate losses and update weights
            loss.backward()
            optimizer.step()

            # Update lr scheduler
            scheduler.step()

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

                train_token_true_positives += token_true_positives
                train_token_false_positives += token_false_positives
                train_token_true_negatives += token_true_negatives
                train_token_false_negatives += token_false_negatives

            # Calculate sequence prediction metrics
            seq_preds = (seq_logits.view(-1) > 0.5).long()
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

            train_seq_true_positives += seq_true_positives
            train_seq_false_positives += seq_false_positives
            train_seq_true_negatives += seq_true_negatives
            train_seq_false_negatives += seq_false_negatives

            train_total_loss += loss.item()
            train_batches += 1
            train_step += 1

            if args.use_wandb:
                if args.mlo_model:
                    wandb.log(
                        {
                            "train_losses/total_loss": loss.item(),
                            "train_losses/sequence_loss": sentence_loss.item(),
                            "train_losses/token_loss": token_loss.item(),
                            "train_losses/regularizer_loss_a": regularizer_loss_a.item(),
                            "train_losses/regularizer_loss_b": regularizer_loss_b.item(),
                            "train_losses/lr": scheduler.get_last_lr()[0],
                            "train_step": train_step,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "train_losses/total_loss": loss.item(),
                            "train_losses/lr": scheduler.get_last_lr()[0],
                            "train_step": train_step,
                        }
                    )

        # Calculate training metrics
        seq_train_accuracy = (train_seq_true_positives + train_seq_true_negatives) / (
            train_seq_true_positives
            + train_seq_false_positives
            + train_seq_true_negatives
            + train_seq_false_negatives
            + 1e-99
        )
        seq_train_precision = train_seq_true_positives / (
            train_seq_true_positives + train_seq_false_positives + 1e-99
        )
        seq_train_recall = train_seq_true_positives / (
            train_seq_true_positives + train_seq_false_negatives + 1e-99
        )
        seq_train_f1 = (2 * seq_train_precision * seq_train_recall) / (
            seq_train_precision + seq_train_recall + 1e-99
        )
        seq_train_f05 = (1.25 * seq_train_precision * seq_train_recall) / (
            0.25 * seq_train_precision + seq_train_recall + 1e-99
        )

        token_train_accuracy = (
            train_token_true_positives + train_token_true_negatives
        ) / (
            train_token_true_positives
            + train_token_false_positives
            + train_token_true_negatives
            + train_token_false_negatives
            + 1e-99
        )
        token_train_precision = train_token_true_positives / (
            train_token_true_positives + train_token_false_positives + 1e-99
        )
        token_train_recall = train_token_true_positives / (
            train_token_true_positives + train_token_false_negatives + 1e-99
        )
        token_train_f1 = (2 * token_train_precision * token_train_recall) / (
            token_train_precision + token_train_recall + 1e-99
        )
        token_train_f05 = (1.25 * token_train_precision * token_train_recall) / (
            0.25 * token_train_precision + token_train_recall + 1e-99
        )

        train_av_loss = train_total_loss / (train_batches + 1e-99)

        if args.use_wandb:
            wandb.log(
                {
                    "train_losses/average_loss": train_av_loss,
                    "train_metrics/sequence_accuracy": seq_train_accuracy,
                    "train_metrics/sequence_precision": seq_train_precision,
                    "train_metrics/sequence_recall": seq_train_recall,
                    "train_metrics/sequence_f1": seq_train_f1,
                    "train_metrics/sequence_f0.5": seq_train_f05,
                    "train_metrics/token_accuracy": token_train_accuracy,
                    "train_metrics/token_precision": token_train_precision,
                    "train_metrics/token_recall": token_train_recall,
                    "train_metrics/token_f1": token_train_f1,
                    "train_metrics/token_f0.5": token_train_f05,
                    "epoch": epoch,
                }
            )

        # Validation loop
        model.eval()
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
                table = wandb.Table(
                    columns=["Input Text", "Predicted Label", "True Label"]
                )

        with torch.no_grad():
            for idx, (sequences, labels, token_labels) in enumerate(val_loader):
                # Zero any accumulated gradients
                optimizer.zero_grad()

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
                token_labels = torch.tensor(
                    token_labels, dtype=torch.float, device=device
                )

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
                    # outputs = model(
                    #     input_ids, attention_mask=attention_masks, labels=labels.long(),
                    # )
                    # loss = outputs.loss
                    # seq_logits = torch.argmax(outputs.logits, dim=1)
                    outputs = model(input_ids, attention_mask=attention_masks, labels=labels.long(),)
                    loss, logits, _ = outputs
                    seq_logits = torch.argmax(logits, dim=1)

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

                    val_token_true_positives += token_true_positives
                    val_token_false_positives += token_false_positives
                    val_token_true_negatives += token_true_negatives
                    val_token_false_negatives += token_false_negatives

                # Calculate sequence prediction metrics
                seq_preds = (seq_logits.view(-1) > 0.5).long()
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

                val_total_loss += loss.item()
                val_batches += 1
                val_step += 1

                if args.use_wandb:
                    if args.mlo_model:
                        wandb.log(
                            {
                                "val_losses/total_loss": loss.item(),
                                "val_losses/sequence_loss": sentence_loss.item(),
                                "val_losses/token_loss": token_loss.item(),
                                "val_losses/regularizer_loss_a": regularizer_loss_a.item(),
                                "val_losses/regularizer_loss_b": regularizer_loss_b.item(),
                                "val_step": val_step,
                            }
                        )
                    else:
                        wandb.log(
                            {
                                "val_losses/total_loss": loss.item(),
                                "val_step": val_step,
                            }
                        )

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
                        true_label = seq_actuals[i].long().item()
                        pred_label = seq_preds[i].long().item()

                        if args.mlo_model:
                            true_token_labels = (
                                token_labels[i][
                                    (token_labels[i] == 0) | (token_labels[i] == 1)
                                ]
                                .long()
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            pred_token_labels = (
                                token_preds[i][
                                    (token_labels[i] == 0) | (token_labels[i] == 1)
                                ]
                                .long()
                                .cpu()
                                .detach()
                                .numpy()
                            )
                            table.add_data(
                                input_text,
                                str(pred_token_labels)[1:-1],
                                str(true_token_labels)[1:-1],
                                str(pred_label),
                                str(true_label),
                            )
                        else:
                            table.add_data(input_text, str(pred_label), str(true_label))
                    wandb.log({f"validation samples: {args.dataset}": table})

        # Epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Calculate validation metrics
        seq_val_accuracy = (val_seq_true_positives + val_seq_true_negatives) / (
            val_seq_true_positives
            + val_seq_false_positives
            + val_seq_true_negatives
            + val_seq_false_negatives
            + 1e-99
        )
        seq_val_precision = val_seq_true_positives / (
            val_seq_true_positives + val_seq_false_positives + 1e-99
        )
        seq_val_recall = val_seq_true_positives / (
            val_seq_true_positives + val_seq_false_negatives + 1e-99
        )
        seq_val_f1 = (2 * seq_val_precision * seq_val_recall) / (
            seq_val_precision + seq_val_recall + 1e-99
        )
        seq_val_f05 = (1.25 * seq_val_precision * seq_val_recall) / (
            0.25 * seq_val_precision + seq_val_recall + 1e-99
        )

        token_val_accuracy = (val_token_true_positives + val_token_true_negatives) / (
            val_token_true_positives
            + val_token_false_positives
            + val_token_true_negatives
            + val_token_false_negatives
            + 1e-99
        )
        token_val_precision = val_token_true_positives / (
            val_token_true_positives + val_token_false_positives + 1e-99
        )
        token_val_recall = val_token_true_positives / (
            val_token_true_positives + val_token_false_negatives + 1e-99
        )
        token_val_f1 = (2 * token_val_precision * token_val_recall) / (
            token_val_precision + token_val_recall + 1e-99
        )
        token_val_f05 = (1.25 * token_val_precision * token_val_recall) / (
            0.25 * token_val_precision + token_val_recall + 1e-99
        )

        val_av_loss = val_total_loss / val_batches

        if args.use_wandb:
            wandb.log(
                {
                    "val_losses/average_loss": val_av_loss,
                    "val_metrics/sequence_accuracy": seq_val_accuracy,
                    "val_metrics/sequence_precision": seq_val_precision,
                    "val_metrics/sequence_recall": seq_val_recall,
                    "val_metrics/sequence_f1": seq_val_f1,
                    "val_metrics/sequence_f0.5": seq_val_f05,
                    "val_metrics/token_accuracy": token_val_accuracy,
                    "val_metrics/token_precision": token_val_precision,
                    "val_metrics/token_recall": token_val_recall,
                    "val_metrics/token_f1": token_val_f1,
                    "val_metrics/token_f0.5": token_val_f05,
                    "epoch": epoch,
                }
            )

        if not args.silent:
            print()
            print("-" * 30)
            print("Epoch {} / {}".format(epoch, args.epochs))
            print("-" * 30)
            print()
            print("Training loss: {:.4f}".format(train_av_loss))
            print("Validation loss: {:.4f}".format(val_av_loss))
            print()
            print("Training sequence accuracy: {:.4f}".format(seq_train_accuracy))
            print("Training sequence precision: {:.4f}".format(seq_train_precision))
            print("Training sequence recall: {:.4f}".format(seq_train_recall))
            print("Training sequence f1: {:.4f}".format(seq_train_f1))
            print("Training sequence f0.5: {:.4f}".format(seq_train_f05))
            print()
            if args.mlo_model:
                print("Training token accuracy: {:.4f}".format(token_train_accuracy))
                print("Training token precision: {:.4f}".format(token_train_precision))
                print("Training token recall: {:.4f}".format(token_train_recall))
                print("Training token f1: {:.4f}".format(token_train_f1))
                print("Training token f0.5: {:.4f}".format(token_train_f05))
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

            print("Learning rate value: {}".format(scheduler.get_last_lr()[0]))
            print("Epoch time: {:.0f}".format(epoch_time))

        if args.early_stopping_objective == "loss":
            objective = -val_av_loss
            best_objective = -best_val_loss
        elif args.early_stopping_objective == "seq_f1":
            objective = seq_val_f1
            best_objective = best_seq_val_f1
        elif args.early_stopping_objective == "token_f1":
            objective = token_val_f1
            best_objective = best_token_val_f1
        elif args.early_stopping_objective == "seq_f05":
            objective = seq_val_f05
            best_objective = best_seq_val_f05
        elif args.early_stopping_objective == "token_f05":
            objective = token_val_f05
            best_objective = best_token_val_f05

        # Determine whether to do early stopping
        if objective > best_objective:
            best_epoch = epoch

            best_train_loss = train_av_loss
            best_seq_train_accuracy = seq_train_accuracy
            best_seq_train_precision = seq_train_precision
            best_seq_train_recall = seq_train_recall
            best_seq_train_f1 = seq_train_f1
            best_seq_train_f05 = seq_train_f05

            best_val_loss = val_av_loss
            best_seq_val_accuracy = seq_val_accuracy
            best_seq_val_precision = seq_val_precision
            best_seq_val_recall = seq_val_recall
            best_seq_val_f1 = seq_val_f1
            best_seq_val_f05 = seq_val_f05

            if args.mlo_model:
                best_token_train_accuracy = token_train_accuracy
                best_token_train_precision = token_train_precision
                best_token_train_recall = token_train_recall
                best_token_train_f1 = token_train_f1
                best_token_train_f05 = token_train_f05

                best_token_val_accuracy = token_val_accuracy
                best_token_val_precision = token_val_precision
                best_token_val_recall = token_val_recall
                best_token_val_f1 = token_val_f1
                best_token_val_f05 = token_val_f05

            no_improvement_num = 0

            if args.save_model:
                torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

        else:
            no_improvement_num += 1

        if no_improvement_num == args.early_stopping_patience:
            break

    training_finish = time.time()
    training_time = training_finish - training_start

    if args.use_wandb:
        if args.mlo_model:
            wandb.log(
                {
                    "best_epoch": best_epoch,
                    "train_summary_metrics/best_average_loss": best_train_loss,
                    "train_summary_metrics/best_sequence_accuracy": best_seq_train_accuracy,
                    "train_summary_metrics/best_sequence_precision": best_seq_train_precision,
                    "train_summary_metrics/best_sequence_recall": best_seq_train_recall,
                    "train_summary_metrics/best_sequence_f1": best_seq_train_f1,
                    "train_summary_metrics/best_sequence_f0.5": best_seq_train_f05,
                    "val_summary_metrics/best_average_loss": best_val_loss,
                    "val_summary_metrics/best_sequence_accuracy": best_seq_val_accuracy,
                    "val_summary_metrics/best_sequence_precision": best_seq_val_precision,
                    "val_summary_metrics/best_sequence_recall": best_seq_val_recall,
                    "val_summary_metrics/best_sequence_f1": best_seq_val_f1,
                    "val_summary_metrics/best_sequence_f0.5": best_seq_val_f05,
                    "train_summary_metrics/best_token_accuracy": best_token_train_accuracy,
                    "train_summary_metrics/best_token_precision": best_token_train_precision,
                    "train_summary_metrics/best_token_recall": best_token_train_recall,
                    "train_summary_metrics/best_token_f1": best_token_train_f1,
                    "train_summary_metrics/best_token_f0.5": best_token_train_f05,
                    "val_summary_metrics/best_token_accuracy": best_token_val_accuracy,
                    "val_summary_metrics/best_token_precision": best_token_val_precision,
                    "val_summary_metrics/best_token_recall": best_token_val_recall,
                    "val_summary_metrics/best_token_f1": best_token_val_f1,
                    "val_summary_metrics/best_token_f0.5": best_token_val_f05,
                    "training_time": training_time,
                }
            )
        else:
            wandb.log(
                {
                    "best_epoch": best_epoch,
                    "train_summary_metrics/best_average_loss": best_train_loss,
                    "train_summary_metrics/best_sequence_accuracy": best_seq_train_accuracy,
                    "train_summary_metrics/best_sequence_precision": best_seq_train_precision,
                    "train_summary_metrics/best_sequence_recall": best_seq_train_recall,
                    "train_summary_metrics/best_sequence_f1": best_seq_train_f1,
                    "train_summary_metrics/best_sequence_f0.5": best_seq_train_f05,
                    "val_summary_metrics/best_average_loss": best_val_loss,
                    "val_summary_metrics/best_sequence_accuracy": best_seq_val_accuracy,
                    "val_summary_metrics/best_sequence_precision": best_seq_val_precision,
                    "val_summary_metrics/best_sequence_recall": best_seq_val_recall,
                    "val_summary_metrics/best_sequence_f1": best_seq_val_f1,
                    "val_summary_metrics/best_sequence_f0.5": best_seq_val_f05,
                    "training_time": training_time,
                }
            )

    if not args.silent:
        print()
        print("-" * 30)
        print("Training ended at epoch {}".format(epoch))
        print("-" * 30)
        print()
        print("Best epoch: {:.4f}".format(best_epoch))
        print()
        print("Best training loss: {:.4f}".format(best_train_loss))
        print("Best validation loss: {:.4f}".format(best_val_loss))
        print()
        print("Best training sequence accuracy: {:.4f}".format(best_seq_train_accuracy))
        print(
            "Best training sequence precision: {:.4f}".format(best_seq_train_precision)
        )
        print("Best training sequence recall: {:.4f}".format(best_seq_train_recall))
        print("Best training sequence f1: {:.4f}".format(best_seq_train_f1))
        print("Best training sequence f0.5: {:.4f}".format(best_seq_train_f05))
        print()
        print("Best validation sequence accuracy: {:.4f}".format(best_seq_val_accuracy))
        print(
            "Best validation sequence precision: {:.4f}".format(best_seq_val_precision)
        )
        print("Best validation sequence recall: {:.4f}".format(best_seq_val_recall))
        print("Best validation sequence f1: {:.4f}".format(best_seq_val_f1))
        print("Best validation sequence f0.5: {:.4f}".format(best_seq_val_f05))
        print()
        if args.mlo_model:
            print(
                "Best training token accuracy: {:.4f}".format(best_token_train_accuracy)
            )
            print(
                "Best training token precision: {:.4f}".format(
                    best_token_train_precision
                )
            )
            print("Best training token recall: {:.4f}".format(best_token_train_recall))
            print("Best training token f1: {:.4f}".format(best_token_train_f1))
            print("Best training token f0.5: {:.4f}".format(best_token_train_f05))
            print()
            print(
                "Best validation token accuracy: {:.4f}".format(best_token_val_accuracy)
            )
            print(
                "Best validation token precision: {:.4f}".format(
                    best_token_val_precision
                )
            )
            print("Best validation token recall: {:.4f}".format(best_token_val_recall))
            print("Best validation token f1: {:.4f}".format(best_token_val_f1))
            print("Best validation token f0.5: {:.4f}".format(best_token_val_f05))
            print()
        print("Training time: {:.0f}".format(training_time))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train model")
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
        "--epochs",
        action="store",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )

    parser.add(
        "--learning_rate",
        action="store",
        type=float,
        default=1e-99,
        help="Learning rate (default: 0.0001)",
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
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug outputs (default: False)",
    )

    parser.add(
        "--early_stopping_patience",
        action="store",
        type=int,
        default=5,
        help="Number of epochs to wait for performance to improve (default: 5)",
    )

    parser.add(
        "--early_stopping_objective",
        action="store",
        type=str,
        default="loss",
        help="Objective metric to determine early stopping by (default: loss)",
    )

    parser.add(
        "--save_model",
        action="store_true",
        default=False,
        help="Save best model (default: False)",
    )

    parser.add(
        "--lr_optimizer",
        action="store",
        type=str,
        default="adamw",
        help="Type of optimizer to use (default: adamw)",
    )

    parser.add(
        "--lr_weight_decay",
        action="store",
        type=float,
        default=0.0,
        help="Optimizer weight decay (default: 0.0)",
    )

    parser.add(
        "--lr_epsilon",
        action="store",
        type=float,
        default=1e-7,
        help="Optimizer weight decay (default: 1e-7)",
    )

    parser.add(
        "--lr_scheduler",
        action="store",
        type=str,
        default="steplr",
        help="Type of shceduler to use (default: steplr)",
    )

    parser.add(
        "--lr_scheduler_step",
        action="store",
        type=int,
        default=5,
        help="Scheduler step size (default: 5)",
    )

    parser.add(
        "--lr_scheduler_gamma",
        action="store",
        type=float,
        default=1,
        help="Scheduler gamma (default: 1)",
    )

    parser.add(
        "--lr_scheduler_warmup_ratio",
        action="store",
        type=float,
        default=0.1,
        help="Scheduler warmup ratio (default: 0.1)",
    )

    parser.add(
        "--soft_attention_beta",
        action="store",
        type=float,
        default=1,
        help="Soft attention beta value (default: 1)",
    )

    parser.add(
        "--sentence_loss_weight",
        action="store",
        type=float,
        default=1,
        help="Sentence loss weight (default: 1)",
    )

    parser.add(
        "--token_loss_weight",
        action="store",
        type=float,
        default=1,
        help="Token loss weight (default: 1)",
    )

    parser.add(
        "--regularizer_loss_weight",
        action="store",
        type=float,
        default=0.01,
        help="Regularizer loss weight (default: 0.01)",
    )

    parser.add(
        "--token_supervision",
        action="store_true",
        default=False,
        help="Use token supervision (default: False)",
    )

    parser.add(
        "--sequence_supervision",
        action="store_true",
        default=False,
        help="Use sequence supervision (default: False)",
    )

    parser.add(
        "--regularization_losses",
        action="store_true",
        default=False,
        help="Use regularization losses (default: False)",
    )

    parser.add(
        "--normalise_supervised_losses",
        action="store_true",
        default=False,
        help="Normalise supervised losses (default: False)",
    )

    parser.add(
        "--normalise_regularization_losses",
        action="store_true",
        default=False,
        help="Normalise regularisation losses (default: False)",
    )

    parser.add(
        "--lr_momentum",
        action="store",
        type=float,
        default=0.9,
        help="Learning rate momentum (default: 0.9)",
    )

    parser.add(
        "--subword_method",
        action="store",
        type=str,
        default="first",
        help="Method for dealing with subwords (default: first)",
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
    train(args)
