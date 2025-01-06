import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_add_pool


#39 
class GraphNN(torch.nn.Module):
    def __init__(self, out_channels, heads, num_layers, edge_feature_dims : tuple = None):
        super(GraphNN, self).__init__() 
        if (out_channels % heads != 0):
            raise ValueError("The number of heads doesn't divide the output dimension.")
        

        self.mg_encoding = torch.nn.LazyLinear(out_channels, bias=False)
        self.pg_encoding = torch.nn.LazyLinear(out_channels, bias=False)
        
        mods = []
        for i in range(num_layers):
            mods.append(GATv2Conv(-1, out_channels=int(out_channels/heads), heads=heads, edge_dim=edge_feature_dims[0]))
            mods.append(torch.nn.Linear(out_channels, out_channels))

        self.mol_gtrans = torch.nn.ModuleList(mods)

        mods = []
        for i in range(num_layers):
            mods.append(GATv2Conv(-1, out_channels=int(out_channels/heads), heads=heads, edge_dim=edge_feature_dims[1]))
            mods.append(torch.nn.Linear(out_channels, out_channels))

        self.prot_gtrans = torch.nn.ModuleList(mods)


        self.fin_fc = torch.nn.ModuleList(
            [
                torch.nn.Linear(out_channels * 2, out_channels),
                torch.nn.ELU(),
                torch.nn.Linear(out_channels, 1),
                torch.nn.Sigmoid(),
            ]
        )
        

    def forward(self, mg, pg):
        mg_x : torch.Tensor = mg.x
        pg_x : torch.Tensor = pg.x
        
        for i in range(len(self.mol_gtrans)/2):
            idx = i * 2
            mg_x = self.mol_gtrans[idx](mg_x, edge_index=mg.edge_index, edge_attr=mg.edge_attr) #GAT
            mg_x_res = self.mol_gtrans[idx + 1](mg_x) #FC
            mg_x_res = torch.nn.functional.elu(mg_x_res)
            mg_x += mg_x_res

        for i in range(len(self.prot_gtrans)/2):
            idx = i * 2
            pg_x = self.prot_gtrans[idx](pg_x, edge_index=pg_x.edge_index, edge_attr=pg_x.edge_attr)
            pg_x_res = self.prot_gtrans[idx + 1](pg_x)
            pg_x_res = torch.nn.functional.elu(pg_x_res)
            pg_x += pg_x_res

        mg_pool = global_add_pool(mg_x)
        pg_pool = global_add_pool(pg_x)

        x = torch.concat((mg_pool, pg_pool), dim=-1)

        for layer in self.fin_fc:
            x = layer(x)
        
        return x
        


if __name__ == "__main__": 
    GraphNN(32, 4, 1, (1, 1))