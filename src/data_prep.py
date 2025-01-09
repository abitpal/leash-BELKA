import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import graphein.protein as gp
import torch
from torch_geometric.data import Data
from functools import partial
import pickle
from tqdm import tqdm
import os
from graphein.ml.conversion import convert_nx_to_pyg_data
import kujira
from sklearn.preprocessing import StandardScaler
from dask import delayed, compute
from dask.diagnostics import ProgressBar


class graph_data:
    def __init__(self, path : str, out_path : str):
        self.df : pd.DataFrame = pd.read_csv(path)
        self.out_path = out_path
        self.BOND_TYPES : list = [val for _, val in rdkit.Chem.rdchem.BondType.values.items()]
        self.BOND_INDEX : dict = {bond : i for i, bond in enumerate(self.BOND_TYPES)}
        self.ATOM_TYPES : list = ['B', 'Dy', 'N', 'O', 'Br', 'S', 'Cl', 'Si', 'C', 'F', 'H', 'I']
        self.ATOM_INDEX : dict = {atom : i for i, atom in enumerate(self.ATOM_TYPES)}
        self.gp_dist_edge_func = {"edge_construction_functions": [partial(gp.add_distance_threshold, threshold=5, long_interaction_threshold=0)]}
        self.gp_one_hot = {"node_metadata_functions": [gp.amino_acid_one_hot]}
        self.gp_config = gp.ProteinGraphConfig(**{**self.gp_dist_edge_func, **self.gp_one_hot})
        self.protein_to_pdb = {"sEH" : "3I28", "HSA" : "1AO6", "BRD4" : "7USK"}
        self.scaler = StandardScaler()
        self.device = "mps"

    # Function to process SMILES strings into a PyG graph
    def smiles_to_pyg(self, smiles: str):
        """
        Converts a SMILES string into a PyTorch Geometric Data object.

        Args:
            smiles (str): SMILES representation of a molecule.

        Returns:
            Data: PyG graph representing the molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        editable_mold = Chem.EditableMol(mol)

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Dy":
                editable_mold.RemoveAtom(atom.GetIdx()) 
            
        mol = editable_mold.GetMol()

        params = AllChem.ETKDG()
        params.maxAttempts = 5000  # Increase embedding attempts
        params.boxSizeMult = 2.0

        AllChem.EmbedMolecule(mol, params)


        # Node features: one-hot encoded atom type and 3D position
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        try: 
            positions = mol.GetConformer().GetPositions()
        except:
            return None

        #normalize positions
        positions -= positions.mean()
        positions /= positions.std()

        atom_indices = [self.ATOM_INDEX.get(atom) for atom in atoms]
        one_hot_atoms = np.eye(len(self.ATOM_TYPES))[atom_indices]
        x = np.hstack([one_hot_atoms, positions])

        # Edge index and features: bond type and distances

        edges = np.array([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()

        bond_types = np.array([self.BOND_INDEX.get(bond.GetBondType()) for bond in mol.GetBonds()])
        bond_types = np.concatenate([bond_types, bond_types], axis=0)
        one_hot_bonds = np.eye(len(self.BOND_TYPES))[bond_types]

        distances = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).reshape(-1, 1)

        edge_attr = np.concatenate([one_hot_bonds, distances], axis=-1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def preprocess_nx_graph(self, graph):
        for node, attrs in graph.nodes(data=True):
            for key, value in attrs.items():
                if not isinstance(value, (list, tuple)):
                    graph.nodes[node][key] = [value]
        for u, v, attrs in graph.edges(data=True):
            for key, value in attrs.items():
                if not isinstance(value, (list, tuple)):
                    graph.edges[u, v][key] = [value]
        return graph
    
    def protein_to_pyg(self, protein: str) -> Data:
        """
        Converts a protein PDB code into a PyTorch Geometric Data object using Graphein.

        Args:
            pdb_code (str): PDB code of the protein.
            config (gp.ProteinGraphConfig): Configuration for protein graph construction.

        Returns:
            Data: PyG graph representing the protein.
        """
        protein_graph = gp.construct_graph(config=self.gp_config, pdb_code=self.protein_to_pdb[protein])
        nx_graph = self.preprocess_nx_graph(protein_graph)
        data = convert_nx_to_pyg_data(nx_graph)

        node_attributes = []
        for _, node_data in nx_graph.nodes(data=True):
            l = list(node_data.values())
            a = l[-1] + l[5]
            node_attributes.append([
                x
                for xs in a
                for x in xs
            ])
            
        # Convert to a PyTorch tensor and assign to data.x
        data.x = torch.tensor(node_attributes, dtype=torch.float, device=self.device)
        data.edge_attr = torch.tensor(data.distance, dtype=torch.float, device=self.device)
        data.edge_attr = (data.edge_attr - data.edge_attr.mean()) / data.edge_attr.std()
        data.edge_attr = data.edge_attr.unsqueeze(1)

        #normalize data
        for i in range(3):
            data.x[:, -i - 1] -= data.x[:, -i - 1].mean()
            data.x[:, -i - 1] /= data.x[:, -i - 1].std()

        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

    def pack(self, protein=True, molecule=True):


        """
        Processes a dataset of small molecules and a protein into PyG graphs.

        Args:
            dataset (pd.DataFrame): DataFrame containing SMILES strings of small molecules.
            pdb_code (str): PDB code of the protein.
            config (gp.ProteinGraphConfig): Configuration for protein graph construction.

        Returns:
            list: List of PyG Data objects for small molecules and the protein graph.
        """

        protein_graphs = {}
        
        if (protein):
            print("Converting proteins to pyg...")
            for protein in tqdm(set(self.df['protein_name'])):
                protein_graphs[protein] = self.protein_to_pyg(protein)

        molecule_graphs = {}
        
        if (molecule):
            # Process small molecules
            def process_smiles(smiles):
                return smiles, self.smiles_to_pyg(smiles)
            
            smiles_list = list(set(self.df['molecule_smiles']))

            print("Creating tasks")
            tasks = [delayed(process_smiles)(smiles) for smiles in smiles_list]

            print("Starting compute")

            with ProgressBar():  # This shows the progress bar for Dask compute
                results = compute(*tasks)

            molecule_graphs = dict(results)

        return protein_graphs, molecule_graphs
    
    
    def __call__(self, protein=True, molecule=True):


        """
        Processes a dataset of small molecules and a protein into PyG graphs.

        Args:
            dataset (pd.DataFrame): DataFrame containing SMILES strings of small molecules.
            pdb_code (str): PDB code of the protein.
            config (gp.ProteinGraphConfig): Configuration for protein graph construction.

        Returns:
            list: List of PyG Data objects for small molecules and the protein graph.
        """

        print("Converting proteins to pyg...")

        pg, mg = self.pack(protein, molecule)

        if (protein):
            with open(os.path.join(self.out_path, 'protein_graph.pkl'), 'wb') as f:
                pickle.dump(pg, f)
                

        if (molecule):
            with open(os.path.join(self.out_path, 'molecule_graph.pkl'), 'wb') as f:
                pickle.dump(mg, f)



if __name__ == "__main__": 
    path = "data/test.csv"
    out_path = "data/main-test-graphs"
    graph_data(path, out_path)() 
