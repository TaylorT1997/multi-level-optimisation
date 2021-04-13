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

    if args.use_wandb:
        wandb.init(project="multi-level-optimisation", entity="taylort1997")
        wandb.config.update(args)
        wandb.watch(model)

    dataset = BinaryTSVDataset(
        dataset_name=args.dataset, tokenizer=tokenizer, mode=args.mode
    )

    model.to(device)
    model.train()

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    optim = AdamW(model.parameters(), lr=args.learning_rate)

    train_step = 0

    for epoch in range(args.epochs):
        for idx, (input_ids, attention_masks, labels) in enumerate(train_loader):
            optim.zero_grad()

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            if args.use_wandb:
                wandb.log(
                    {
                        "mse_loss_train": loss.item(),
                        "epoch": epoch,
                        "train_step": train_step,
                    }
                )

                if idx == 0:
                    table = wandb.Table(
                        columns=["Input Text", "Predicted Label", "True Label"]
                    )
                    for i in range(args.batch_size):
                        input_text = tokenizer.decode(
                            input_ids[i],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        true_label = labels[i].detach().numpy()
                        pred_label = outputs.logits[i].detach().numpy()[0]
                        table.add_data(input_text, str(pred_label), str(true_label))

                    wandb.log({"{}_examples".format(args.dataset): table})

            train_step += 1

    model.eval()


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
        "-s",
        "--mode",
        action="store",
        default="dev",
        help="Dataset split to train on (train/dev/test)",
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
