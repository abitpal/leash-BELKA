import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv

class GraphNN(torch.nn.Module):
    def __init__(self, out_channels, heads, num_layers, edge_feature_dims : tuple = None):
        super(GraphNN, self).__init__() 
        if (out_channels % heads != 0):
            raise ValueError("The number of heads doesn't divide the output dimension.")
        
        mods = []
        for i in range(num_layers):
            mods.append(GATv2Conv(-1, out_channels=int(out_channels/heads), heads=heads, edge_dim=edge_feature_dims[0]))
            mods.append(torch.nn.ELU())

        self.molecule_GAT = torch.nn.ModuleList(mods)

        mods = []
        for i in range(num_layers):
            mods.append(GATv2Conv(-1, out_channels=int(out_channels/heads), heads=heads, edge_dim=edge_feature_dims[1]))
            mods.append(torch.nn.ELU())

        self.protein_GAT = torch.nn.ModuleList(mods)

    def forward(self, mg, pg):
        pass


if __name__ == "__main__": 
    GraphNN(32, 4, 1, (1, 1))