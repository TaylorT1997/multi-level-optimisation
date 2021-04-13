import sys
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig

from data_loading.datasets import BinaryTSVDataset

import wandb


def collate_fn(batch):
    input_ids, attention_masks, labels = [], [], []
    for data_dict, label in batch:
        input_ids.append(data_dict["input_ids"])
        attention_masks.append(data_dict["attention_mask"])
        labels.append(label)

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    return input_ids, attention_masks, labels


def train(args):
    torch.manual_seed(666)
    print(vars(args))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_config = BertConfig.from_pretrained(args.model, num_labels=1)
    model = BertForSequenceClassification(model_config)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=1, bias=True), torch.nn.Sigmoid()
    )

    model.to(device)

    train_dataset = BinaryTSVDataset(
        dataset_name=args.dataset, tokenizer=tokenizer, mode="train"
    )
    val_dataset = BinaryTSVDataset(
        dataset_name=args.dataset, tokenizer=tokenizer, mode="dev"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    optim = AdamW(model.parameters(), lr=args.learning_rate)
    train_step = 0
    val_step = 0

    # Log and print configuration
    if args.use_wandb:
        wandb.init(project="multi-level-optimisation", entity="taylort1997")
        wandb.config.update(args)
        wandb.watch(model)

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        train_corrects = 0
        train_samples = 0
        train_total_loss = 0
        train_batches = 0

        for idx, (input_ids, attention_masks, labels) in enumerate(train_loader):
            optim.zero_grad()

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            preds = outputs.logits > 0.5
            corrects = preds.view(-1) == labels

            train_corrects += torch.sum(corrects).detach().numpy()
            train_samples += len(corrects)
            train_total_loss += loss.item()
            train_batches += 1
            train_step += 1

            if args.use_wandb:
                wandb.log({"step_loss_train": loss.item(), "train_step": train_step})

        train_accuracy = train_corrects / train_samples
        train_av_loss = train_total_loss / train_batches

        if args.use_wandb:
            wandb.log(
                {
                    "epoch_loss_train": train_av_loss,
                    "epoch_accuracy_train": train_accuracy,
                    "epoch": epoch,
                }
            )

        # Validation loop
        model.eval()
        val_corrects = 0
        val_samples = 0

        if args.use_wandb:
            table = wandb.Table(columns=["Input Text", "Predicted Label", "True Label"])

        for idx, (input_ids, attention_masks, labels) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss

            preds = outputs.logits > 0.5
            corrects = preds.view(-1) == labels

            val_corrects += torch.sum(corrects).detach().numpy()
            val_samples += len(corrects)
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
                    true_label = labels[i].detach().numpy()
                    pred_label = outputs.logits[i].detach().numpy()[0]
                    table.add_data(input_text, str(pred_label), str(true_label))

        val_accuracy = val_corrects / val_samples
        val_av_loss = val_total_loss / val_batches

        if args.use_wandb:
            wandb.log({"dev_set_examples".format(args.dataset): table})
            wandb.log(
                {
                    "epoch_loss_val": val_av_loss,
                    "epoch_accuracy_val": val_accuracy,
                    "epoch": epoch,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "-m",
        "--model",
        action="store",
        default="bert-base-cased",
        help="Pretrained model to use",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        action="store",
        default="bert-base-cased",
        help="Pretrained tokenizer to use",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        default="conll_10",
        help="Dataset to train on",
    )

    parser.add_argument(
        "-b", "--batch_size", action="store", default=8, help="Batch size (default: 8)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        default=10,
        help="Number of epochs (default: 10)",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        action="store",
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

    args = parser.parse_args()

    train(args)
