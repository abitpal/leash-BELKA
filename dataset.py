import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, df, device = None):
        self.df = df
        self.molecule_graphs = pickle.load(open("data/graphs/molecule_graph.pkl", "rb"))
        self.protein_graphs = pickle.load(open("data/graphs/protein_graph.pkl", "rb"))
        self.device = "mps"
        if (device):
            self.device = device
            

    def __len__(self):
        return len(self.df)
    
    def toTensor(self, val):
        return torch.Tensor(val, dtype=torch.float).to(self.device)

    def __getitem__(self, idx):
        #molecule_smiles,protein_name,binds
        molecule_smiles = self.df['molecule_smiles'][idx]
        protein_name = self.df['protein_name'][idx]
        binds = torch.Tensor(self.df['binds'][idx]).to(self.device)
        
        mg = self.molecule_graphs.get(molecule_smiles)
        pg = self.protein_graphs.get(protein_name)

        mg.x = self.toTensor(mg.x)
        pg.x = self.toTensor(pg.x)

        mg.edge_attr = self.toTensor(mg.edge_attr)
        pg.edge_attr = self.toTensor(pg.edge_attr)

        mg.edge_index = self.toTensor(mg.edge_index)
        pg.edge_index = self.toTensor(pg.edge_index)


        return mg, pg, binds
    
