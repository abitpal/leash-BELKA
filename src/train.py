from dataset import GraphDataset
import pandas as pd
from model import GraphNN
import torch
from torch.utils.data import DataLoader
import kujira
import sys
import wandb

def train(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, tracker = None, epochs=10):
    model.train()
    for epoch in range(epochs): 
        for i, data in enumerate(dataloader): 

            optimizer.zero_grad()

            mg, pg, bind = data
            
            out = model(mg, pg).to("mps")
            loss = criterion(out, bind)

            loss.backward()
            optimizer.step()

            tracker("train", locals())
            wandb.log({"Batch Loss": loss.item(), "Batch": i + epoch * len(dataloader)})

    return model


if __name__ == "__main__":
    train_df = pd.read_csv("data/normalized_train.csv")
    test_df = pd.read_csv("data/normalized_test.csv")
    
    train_dataset = GraphDataset(train_df)
    test_dataset = GraphDataset(test_df)

    print(train_dataset[0][-1])

    batch_size = 20

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=GraphDataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=GraphDataset.collate)

    mg_samp, pg_samp, _ = train_dataset[0]

    model = GraphNN(out_channels=32, heads=4, num_layers=10, edge_feature_dims=(mg_samp.edge_attr.shape[-1], pg_samp.edge_attr.shape[-1])).to('mps')

    tracker = None

    if (len(sys.argv) > 1):
        tracker_path = sys.argv[1]
        tracker = kujira.init(tracker_path)

    criterion = torch.nn.BCELoss().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    wandb.init(project="Leash-BELKA", name="batch-loss-tracking")

    train(model, training_dataloader, criterion, optimizer, tracker)

    wandb.finish()




