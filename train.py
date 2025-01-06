from .dataset import GraphDataset
import pandas as pd
from model import GraphNN
import torch
from torch.utils.data import DataLoader
import kujira
import sys

def train(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker = None):
    model.train()
    for i, data in dataloader: 

        optimizer.zero_grad()

        mg, pg, bind = data
        
        out = model(mg, pg)
        loss = criterion(out, bind)

        loss.backward()
        optimizer.step()

        tracker(**locals())

    return model


if __name__ == "__main__":
    train_df = pd.read_csv("data/normalized_training.csv")
    test_df = pd.read_csv("data/normalized_test.csv")
    
    train_dataset = GraphDataset(train_df)
    test_dataset = GraphDataset(test_df)

    batch_size = 20

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    mg_samp, pg_samp, _ = train_dataset[0]

    model = GraphNN(128, 4, 10, (mg_samp.edge_attr.shape[-1], pg_samp.edge_attr.shape[-1]))

    tracker = None

    if (len(sys.argv) > 1):
        tracker_path = sys.argv[1]
        tracker = kujira.init(tracker_path)

    train(training_dataloader, tracker)




