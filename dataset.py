import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.molecule_graphs = pickle.load(open("data/graphs/molecule_graph.pkl", "rb"))
        self.protein_graphs = pickle.load(open("data/graphs/protein_graph.pkl", "rb"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #molecule_smiles,protein_name,binds
        molecule_smiles = self.df['molecule_smiles'][idx]
        protein_name = self.df['protein_name'][idx]
        binds = self.df['binds'][idx]
        return self.molecule_graphs.get(molecule_smiles), self.protein_graphs.get(protein_name), binds
    
