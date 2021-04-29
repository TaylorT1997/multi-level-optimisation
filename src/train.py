import sys
import argparse
import time
import datetime
import os

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
)

from data_loading.datasets import BinarySentenceTSVDataset, BinaryTokenTSVDataset
from models.model import TokenModel

import wandb


def collate_fn(batch):
    input_ids, attention_masks, labels, token_labels = [], [], [], []
    for data_dict, label, token_label in batch:
        input_ids.append(data_dict["input_ids"])
        attention_masks.append(data_dict["attention_mask"])
        labels.append(label)
        token_labels.append(token_label)

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    return input_ids, attention_masks, labels, token_labels


def train(args):
    torch.manual_seed(666)

    if not args.silent:
        print("*" * 30)
        print(
            "Training: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        print("*" * 30)
        print()
        print("Model: {}".format(args.model))
        print("Tokenizer: {}".format(args.tokenizer))
        print("Dataset: {}".format(args.dataset))
        print()
        print("Batch size: {}".format(args.batch_size))
        print("Epochs: {}".format(args.epochs))
        print("Learning rate: {}".format(args.learning_rate))
        print("Early stopping patience: {}".format(args.early_stopping_patience))
        print()
        print("Optimizer: {}".format(args.lr_optimizer))
        print("Scheduler: {}".format(args.lr_scheduler))
        print("Scheduler step size: {}".format(args.lr_scheduler_step))
        print("Scheduler gamma: {}".format(args.lr_scheduler_gamma))

    if args.save_model:
        model_dir = os.path.join(args.root, "models", "{}_{}".format(args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if "bert-base" in args.model:
        if args.mlo_model:
            model = TokenModel(pretrained_model=args.model)
        else:
            model_config = BertConfig.from_pretrained(args.model, num_labels=1)
            model = BertForSequenceClassification(model_config)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    elif "deberta-base" in args.model:
        if args.mlo_model:
            model = TokenModel(pretrained_model=args.model)
        else:
            model_config = DebertaConfig.from_pretrained(args.model, num_labels=1)
            model = DebertaForSequenceClassification(model_config)
        tokenizer = DebertaTokenizer.from_pretrained(args.tokenizer)

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=1, bias=True), torch.nn.Sigmoid()
    )

    model.to(device)

    train_dataset = BinaryTokenTSVDataset(
        dataset_name=args.dataset, tokenizer=tokenizer, root_dir=args.root, mode="train"
    )
    val_dataset = BinaryTokenTSVDataset(
        dataset_name=args.dataset, tokenizer=tokenizer, root_dir=args.root, mode="dev"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Optimizer and scheduler
    if args.lr_optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.lr_scheduler == "steplr":
        scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    # Record steps
    train_step = 0
    val_step = 0

    # Record best metrics
    best_train_loss = 1e10
    best_train_accuracy = 0
    best_train_precision = 0
    best_train_recall = 0
    best_train_f1 = 0

    best_val_loss = 1e10
    best_val_accuracy = 0
    best_val_precision = 0
    best_val_recall = 0
    best_val_f1 = 0

    # Early stopping
    no_improvement_num = 0

    # Log and print configuration
    if args.use_wandb:
        wandb.init(project="multi-level-optimisation", entity="taylort1997")
        wandb.config.update(args)
        wandb.watch(model)

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training loop
        epoch_start = time.time()
        model.train()

        train_samples = 0
        train_total_loss = 0
        train_batches = 0

        train_true_positives = 0
        train_false_positives = 0
        train_true_negatives = 0
        train_false_negatives = 0

        for idx, (input_ids, attention_masks, labels, token_labels) in enumerate(
            train_loader
        ):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)
            
            if args.mlo_model:
                token_labels = torch.tensor(token_labels, dtype=torch.float, device=device)
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=token_labels
                )
            else:
                outputs = model(
                    input_ids, attention_mask=attention_masks, labels=labels
                )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            preds = [1 if x > 0.5 else 0 for x in outputs.logits.view(-1)]
            actuals = labels.detach().cpu().numpy()

            num_samples = len(actuals)

            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for i in range(num_samples):
                if actuals[i] == preds[i] == 1:
                    true_positives += 1
                if preds[i] == 1 and actuals[i] != preds[i]:
                    false_positives += 1
                if actuals[i] == preds[i] == 0:
                    true_negatives += 1
                if preds[i] == 0 and actuals[i] != preds[i]:
                    false_negatives += 1

            train_true_positives += true_positives
            train_false_positives += false_positives
            train_true_negatives += true_negatives
            train_false_negatives += false_negatives

            train_samples += num_samples
            train_total_loss += loss.item()
            train_batches += 1
            train_step += 1

            if args.use_wandb:
                wandb.log({"step_loss_train": loss.item(), "train_step": train_step})

        # Calculate training metrics
        train_accuracy = (train_true_positives + train_true_negatives) / train_samples
        train_precision = train_true_positives / (
            train_true_positives + train_false_positives + 1e-5
        )
        train_recall = train_true_positives / (
            train_true_positives + train_false_negatives + 1e-5
        )
        train_f1 = (2 * train_precision * train_recall) / (
            train_precision + train_recall + 1e-5
        )
        train_av_loss = train_total_loss / train_batches

        if args.use_wandb:
            wandb.log(
                {
                    "epoch_loss_train": train_av_loss,
                    "epoch_accuracy_train": train_accuracy,
                    "epoch_precision_train": train_precision,
                    "epoch_recall_train": train_recall,
                    "epoch_f1_train": train_f1,
                    "epoch": epoch,
                }
            )

        # Validation loop
        model.eval()
        val_samples = 0
        val_total_loss = 0
        val_batches = 0

        val_true_positives = 0
        val_false_positives = 0
        val_true_negatives = 0
        val_false_negatives = 0

        if args.use_wandb:
            table = wandb.Table(columns=["Input Text", "Predicted Label", "True Label"])

        with torch.no_grad():
            for idx, (input_ids, attention_masks, labels, token_labels) in enumerate(
                val_loader
            ):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = torch.tensor(labels, dtype=torch.float, device=device)

                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss

                preds = [1 if x > 0.5 else 0 for x in outputs.logits.view(-1)]
                actuals = labels.detach().cpu().numpy()

                num_samples = len(actuals)

                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0

                for i in range(num_samples):
                    if actuals[i] == preds[i] == 1:
                        true_positives += 1
                    if preds[i] == 1 and actuals[i] != preds[i]:
                        false_positives += 1
                    if actuals[i] == preds[i] == 0:
                        true_negatives += 1
                    if preds[i] == 0 and actuals[i] != preds[i]:
                        false_negatives += 1

                val_true_positives += true_positives
                val_false_positives += false_positives
                val_true_negatives += true_negatives
                val_false_negatives += false_negatives

                val_samples += num_samples
                val_total_loss += loss.item()
                val_batches += 1
                val_step += 1

                if args.use_wandb:
                    wandb.log(
                        {"step_loss_val": loss.item(), "val_step": val_step,}
                    )

                    for i in range(len(labels)):
                        input_text = tokenizer.decode(
                            input_ids[i],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        true_label = labels[i].detach().cpu().numpy()
                        pred_label = outputs.logits[i].detach().cpu().numpy()
                        table.add_data(input_text, str(pred_label), str(true_label))

        # Epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Calculate validation metrics
        val_accuracy = (val_true_positives + val_true_negatives) / val_samples
        val_precision = val_true_positives / (
            val_true_positives + val_false_positives + 1e-5
        )
        val_recall = val_true_positives / (
            val_true_positives + val_false_negatives + 1e-5
        )
        val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall + 1e-5)
        val_av_loss = val_total_loss / val_batches

        if args.use_wandb:
            wandb.log(
                {
                    "examples_val": table,
                    "epoch_loss_val": val_av_loss,
                    "epoch_accuracy_val": val_accuracy,
                    "epoch_precision_val": val_precision,
                    "epoch_recall_val": val_recall,
                    "epoch_f1_val": val_f1,
                    "epoch_time": epoch_time,
                    "epoch": epoch,
                }
            )

        # Update lr scheduler
        scheduler.step()

        if not args.silent:
            print()
            print("-" * 30)
            print("Epoch {} / {}".format(epoch, args.epochs))
            print("-" * 30)
            print()
            print("Training loss: {:.4f}".format(train_av_loss))
            print("Training accuracy: {:.2f}".format(train_accuracy))
            print("Training precision: {:.2f}".format(train_precision))
            print("Training recall: {:.2f}".format(train_recall))
            print("Training f1: {:.2f}".format(train_f1))
            print()
            print("Validation loss: {:.4f}".format(val_av_loss))
            print("Validation accuracy: {:.2f}".format(val_accuracy))
            print("Validation precision: {:.2f}".format(val_precision))
            print("Validation recall: {:.2f}".format(val_recall))
            print("Validation f1: {:.2f}".format(val_f1))
            print()
            print("Epoch time: {:.0f}".format(epoch_time))

        # Determine whether to do early stopping
        if val_av_loss < best_val_loss:
            best_train_loss = train_av_loss
            best_train_accuracy = train_accuracy
            best_train_precision = train_precision
            best_train_recall = train_recall
            best_train_f1 = train_f1

            best_val_loss = val_av_loss
            best_val_accuracy = val_accuracy
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1

            no_improvements_num = 0

            if args.save_model:
                torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            
        else:
            no_improvement_num += 1
        
        if no_improvement_num == args.early_stopping_patience:
            break

    training_finish = time.time()
    training_time = training_finish - training_start

    if not args.silent:
        print()
        print("-" * 30)
        print("Training ended at epoch {}".format(epoch))
        print("-" * 30)
        print()
        print("Best training loss: {:.4f}".format(best_train_loss))
        print("Best training accuracy: {:.2f}".format(best_train_accuracy))
        print("Best training precision: {:.2f}".format(best_train_precision))
        print("Best training recall: {:.2f}".format(best_train_recall))
        print("Best training f1: {:.2f}".format(best_train_f1))
        print()
        print("Best validation loss: {:.4f}".format(best_val_loss))
        print("Best validation accuracy: {:.2f}".format(best_val_accuracy))
        print("Best validation precision: {:.2f}".format(best_val_precision))
        print("Best validation recall: {:.2f}".format(best_val_recall))
        print("Best validation f1: {:.2f}".format(best_val_f1))
        print()
        print("Training time: {:.0f}".format(training_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument(
        "-r",
        "--root",
        action="store",
        type=str,
        default="/home/tom/Projects/multi-level-optimisation/",
        help="Root directory of project",
    )

    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="bert-base-cased",
        help="Pretrained model to use",
    )

    parser.add_argument(
        "--mlo_model",
        action="store_true",
        default=False,
        help="Use multi-level optimisation model (default: False)",
    )

    parser.add_argument(
        "--tokenizer",
        action="store",
        type=str,
        default="bert-base-cased",
        help="Pretrained tokenizer to use",
    )

    parser.add_argument(
        "--dataset",
        action="store",
        type=str,
        default="conll_10",
        help="Dataset to train on",
    )

    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )

    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )

    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        default=1e-5,
        help="Learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "-w",
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use wandb to track run (default: False)",
    )

    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        default=False,
        help="Silence console prints (default: False)",
    )

    parser.add_argument(
        "--early_stopping_patience",
        action="store",
        type=int,
        default=5,
        help="Number of epochs to wait for performance to improve (default: 5)",
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save best model (default: False)",
    )

    parser.add_argument(
        "--lr_optimizer",
        action="store",
        type=str,
        default="adamw",
        help="Type of optimizer to use (default: adamw)",
    )

    parser.add_argument(
        "--lr_scheduler",
        action="store",
        type=str,
        default="steplr",
        help="Type of shceduler to use (default: steplr)",
    )

    parser.add_argument(
        "--lr_scheduler_step",
        action="store",
        type=int,
        default=5,
        help="Scheduler step size (default: 5)",
    )

    parser.add_argument(
        "--lr_scheduler_gamma",
        action="store",
        type=float,
        default=0.1,
        help="Scheduler gamma (default: 0.1)",
    )

    args = parser.parse_args()

    train(args)
