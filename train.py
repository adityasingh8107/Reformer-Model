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

# Load the pretrained Reformer model and tokenizer
model_name = "google/reformer-crime-and-punishment"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=350)

# Check if the tokenizer has a padding token
if tokenizer.pad_token is None:
    # If tokenizer doesn't have a padding token, set it to the end-of-sequence token (eos_token)
    tokenizer.pad_token = tokenizer.eos_token

# Set the padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = ReformerForSequenceClassification.from_pretrained(model_name, num_labels=2)  # binary classification for sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the data preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")

# Preprocess the dataset
dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into train and test sets
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Define the training arguments
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

# Use the train function to train the model
train(model, train_dataset, test_dataset, training_args.num_train_epochs, learning_rate, loss_fn, optimizer, device)

print("Training complete!")

