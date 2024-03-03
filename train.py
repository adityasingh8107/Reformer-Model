#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trainer import compute_metrics, train
from Model import ReformerForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

dataset = load_dataset("imdb")
data = dataset['train']['test']
np.random.shuffle(dataset)

model_name = "google/reformer-crime-and-punishment"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=350)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = ReformerForSequenceClassification.from_pretrained(model_name, num_labels=2)  # binary classification for sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess_function, batched=True)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

batch_size = 32
num_batches = len(train_dataset) // batch_size

learning_rate = 2e-5  # Adjust as needed
loss_fn = torch.nn.CrossEntropyLoss()  # For binary classification
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Choose suitable optimizer

for i in range(num_batches):
    batch_data = train_dataset[i * batch_size: (i + 1) * batch_size]
    train(model, batch_data, test_dataset, 5, learning_rate, loss_fn, optimizer, device)

print("Training complete!")

