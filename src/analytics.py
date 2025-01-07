import matplotlib.pyplot as plt
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from dataset import GraphDataset


test_df = pd.read_csv("data/normalized_test.csv")

test_dataset = GraphDataset(test_df)

test_dataloader = DataLoader(test_dataset, batch_size=20, collate_fn=GraphDataset.collate)
test_iter = iter(test_dataloader)

def validation_loss(data, model, criterion):
    mg, pg, bind = data
    out = model(mg, pg).to("mps")
    loss = criterion(out, bind)
    return loss

def accuracy(data, model):
    mg, pg, bind = data
    out = model(mg, pg).to("mps")
    out = torch.round(out)
    return (torch.sum(out == bind)/len(bind)).item()



def collect(_type, var):
    if _type == "train" and var['i']%10 == 0:
        loss = var['loss']
        if (var['i'] == 0):
            print("EPOCH: " + str(var['epoch']))

        try:
            val_data = next(test_iter)
        except:
            val_data = next(iter(test_dataset))

        training_loss = loss.item()
        val_loss = validation_loss(val_data, var['model'], var['criterion']).item()
        val_accuracy = accuracy(val_data, var['model'])

        print()

        print("Output mean: " + str(var['out'].mean().item()))
        print("Training loss: " + str(training_loss))
        print("Validation loss: " + str(val_loss))
        print("Validation accuracy: " + str(val_accuracy))

    

        
    