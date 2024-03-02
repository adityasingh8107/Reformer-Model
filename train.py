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

dataset = load_dataset("sst")

model_name = "google/reformer-crime-and-punishment"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=350)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = ReformerForSequenceClassification.from_pretrained(model_name, num_labels=2)  # binary classification for sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess_function, batched=True)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

training_args = TrainingArguments(
    output_dir="./output",  # Specify the output directory
    per_device_train_batch_size=2,  # Decrease the batch size for training
    per_device_eval_batch_size=2,  # Decrease the batch size for evaluation
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

learning_rate = 2e-5  # Adjust as needed
loss_fn = torch.nn.CrossEntropyLoss()  # For binary classification
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Choose suitable optimizer

train(model, train_dataset, test_dataset, training_args.num_train_epochs, learning_rate, loss_fn, optimizer, device)

print("Training complete!")

