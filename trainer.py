#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from Model import ReformerForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Define the evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Define the training loop
def train(model, train_data, test_data, epochs, learning_rate, loss_fn, optimizer, device):
    """
    This function trains a model on the provided data for a specified number of epochs.

    Args:
        model: The model to train.
        train_data: The training data loader.
        test_data: The testing data loader.
        epochs: The number of epochs to train for.
        learning_rate: The learning rate to use for training.
        loss_fn: The loss function to use for calculating the loss.
        optimizer: The optimizer to use for updating the model weights.
        device: The device to use for training (CPU or GPU).
    """

    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training loop
        model.train()
        for batch_idx, batch in enumerate(train_data):
            data, labels = batch["sentence"], batch["label"]
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress (optional)
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_data)} - Loss: {loss.item():.4f}")

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            # Evaluate on test data and calculate metrics (e.g., accuracy)
            test_loss = 0
            correct = 0
            for batch in test_data:
                data, labels = batch["sentence"], batch["label"]
                data, labels = data.to(device), labels.to(device)
                predictions = model(data)
                test_loss += loss_fn(predictions, labels).item()
                correct += (torch.argmax(predictions, dim=-1) == labels).sum().item()
            test_loss /= len(test_data)
            test_acc = correct / len(test_dataset)

learning_rate = 2e-5  # Adjust as needed
loss_fn = torch.nn.CrossEntropyLoss()  # For binary classification
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Choose suitable optimizer

# Use the train function to train the model
train(model, train_dataset, test_dataset, training_args.num_train_epochs, learning_rate, loss_fn, optimizer, device)

print("Training complete!")


# In[ ]:




