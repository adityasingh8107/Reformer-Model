#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from Model import ReformerForSequenceClassification
from sklearn.metrics import accuracy_score

model_name = "google/reformer-crime-and-punishment"
model = ReformerForSequenceClassification.from_pretrained(model_name, num_labels=2)
batch_size = 1

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train(model, train_data, test_data, epochs, learning_rate, loss_fn, optimizer, device):
    num_batches = len(train_data) // batch_size

    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(num_batches):

        

            model.train()
            batch = train_data[i * batch_size: (i + 1) * batch_size]

            data, labels = batch["sentence"], batch["label"]
            data, labels = data.to(device), labels.to(device)

            predictions = model(data)
            loss = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if batch_idx % 100 == 0:
                #print(f"Batch {batch_idx+1}/{len(train_data)} - Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
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

# In[ ]:




