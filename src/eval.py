import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from dataset import GraphDataset
import pandas as pd
from model import GraphNN
import torch
from torch.utils.data import DataLoader
import kujira
import sys
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def eval_loop(model: torch.nn.Module, dataloader: DataLoader, tracker = None):

    results = {'output': np.array([]), 'truth': np.array([])}

    model.train(False)

    model = model.to("mps")

    for i, data in tqdm(enumerate(dataloader)): 
        with torch.no_grad(): 
            mg, pg, bind = data
            out = model(mg, pg).cpu().numpy()
            bind = bind.cpu().numpy()
            results['output'] = np.append(results['output'], out)
            results['truth'] = np.append(results['truth'], bind)
            if (tracker):
                tracker("test", locals())

    return results


def pred(model: torch.nn.Module, dataloader: DataLoader, tracker = None):
    results = np.array([])

    model.train(False)

    for i, data in tqdm(enumerate(dataloader)):  
        with torch.no_grad():
            mg, pg = data
            out = model(mg, pg).cpu().numpy()
            results = np.append(results, out)
            if not (tracker is None):
                tracker("test", ls())

    return results


def eval(test_path, res_path, tracker=None):
    test_df = pd.read_csv(test_path)
    test_dataset = GraphDataset(test_df)

    print(test_dataset[0][-1])

    batch_size = 40

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=GraphDataset.collate)

    mg_samp, pg_samp, _ = test_dataset[0]

    model = GraphNN(out_channels=32, heads=4, num_layers=10, edge_feature_dims=(mg_samp.edge_attr.shape[-1], pg_samp.edge_attr.shape[-1]))
    model.load_state_dict(torch.load("models/gnn-01-06.zip", weights_only=False))

    res = eval_loop(model, test_dataloader, tracker)

    with open(res_path, "wb") as f:
        pickle.dump(res, f)

def get_metrics(res_path):
    with open(res_path, "rb") as f:
        res = pickle.load(f)

    outputs = res['output']
    truths = res['truth']

    # Binarize outputs (assuming threshold of 0.5)
    predictions = (outputs >= 0.5).astype(int)

    accuracy = accuracy_score(truths, predictions)
    precision = precision_score(truths, predictions)
    recall = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    roc_auc = roc_auc_score(truths, outputs)
    conf_matrix = confusion_matrix(truths, predictions)
    conf_matrix = conf_matrix / conf_matrix.sum()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.show()

    plt.savefig("results/metrics/test_confusion_matrix.png")

    return metrics

if __name__ == "__main__":
    if (len(sys.argv) == 3):
        eval(sys.argv[1], sys.argv[2])
    else:
        metrics = get_metrics(sys.argv[1])





