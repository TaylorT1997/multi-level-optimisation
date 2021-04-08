import sys

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig

from data_loading.datasets import BinaryTSVDataset

torch.manual_seed(666)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


config = BertConfig.from_pretrained("bert-base-cased", num_labels=1)

model = BertForSequenceClassification(config)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = BinaryTSVDataset(dataset_name="conll_10", tokenizer=tokenizer, mode="dev")

model.to(device)
model.train()


def collate_fn(batch):
    input_ids, attention_masks, labels = [], [], []
    for data_dict, label in batch:
        input_ids.append(data_dict["input_ids"])
        attention_masks.append(data_dict["attention_mask"])
        labels.append(label)
    return input_ids, attention_masks, labels


train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for idx, (input_ids, attention_masks, labels) in enumerate(train_loader):
        optim.zero_grad()

        input_ids = input_ids[0].to(device)
        attention_masks = attention_masks[0].to(device)
        labels = torch.tensor(labels, dtype=torch.float, device=device)

        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        print(outputs)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
