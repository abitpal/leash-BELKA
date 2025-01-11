import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from data_prep import graph_data
import pandas as pd
from model import GraphNN
from torch.utils.data import DataLoader
from dataset import GraphDataset
import torch
from tqdm import tqdm
from eval import pred
import os

#Process and output results in chunks

# Read the test.parquet file into a pandas DataFrame
test_file = "data/test.csv"
output_file = "results/test/output.csv"

model = None

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

for test_df in tqdm(pd.read_csv(test_file, chunksize=30000)):

    test_df.reset_index(drop=True, inplace=True)

    protein_graphs, molecule_graphs = graph_data(dataframe=test_df).pack()

    test_dataset = GraphDataset(test_df, training=False, molecule_graphs=molecule_graphs, protein_graphs=protein_graphs)
    batch_size = 300

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=GraphDataset.collate_test)

    mg_samp, pg_samp = test_dataset[0]

    if (model == None):
        model = GraphNN(out_channels=32, heads=4, num_layers=10, edge_feature_dims=(mg_samp.edge_attr.shape[-1], pg_samp.edge_attr.shape[-1]))
        model.load_state_dict(torch.load("models/gnn-01-06.zip", weights_only=False))
        model = model.to("mps")

    res = pred(model, test_dataloader)
    
    output_df = pd.DataFrame({'id': test_df['id'], 'binds': res})

    # Save the output DataFrame to a CSV file (mode='a' appends the values)
    output_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
