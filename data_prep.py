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


class graph_data:
    def __init__(self, path : str, out_path : str):
        self.df : pd.DataFrame = pd.read_csv(path)
        self.out_path = out_path
        self.BOND_TYPES : list = [val for key, val in rdkit.Chem.rdchem.BondType.values.items()]
        self.ATOM_TYPES : list = ['B', 'Dy', 'N', 'O', 'Br', 'S', 'Cl', 'Si', 'C', 'F', 'H']
        self.gp_dist_edge_func = {"edge_construction_functions": [partial(gp.add_distance_threshold, threshold=5, long_interaction_threshold=0)]}
        self.gp_one_hot = {"node_metadata_functions": [gp.amino_acid_one_hot]}
        self.gp_config = gp.ProteinGraphConfig(**{**self.gp_dist_edge_func, **self.gp_one_hot})
        self.protein_to_pdb = {"sEH" : "3I28", "HSA" : "1AO6", "BRD4" : "7USK"}

    # Function to process SMILES strings into a PyG graph
    def smiles_to_pyg(self, smiles: str) -> Data:
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

        AllChem.EmbedMolecule(mol, AllChem.ETKDG())


        # Node features: one-hot encoded atom type and 3D position
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = mol.GetConformer().GetPositions()
        x = torch.tensor(
            [np.concatenate([np.eye(len(self.ATOM_TYPES))[self.ATOM_TYPES.index(atom)], positions[i]]) 
            for i, atom in enumerate(atoms)],
            dtype=torch.float
        )

        # Edge index and features: bond type and distances
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_type = bond.GetBondType()
            bond_attr = [1 if bond_type == b else 0 for b in self.BOND_TYPES]
            distance = np.linalg.norm(positions[i] - positions[j])
            edge_attr.append(bond_attr + [distance])
            edge_attr.append(bond_attr + [distance])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

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
        data = convert_nx_to_pyg_data(self.preprocess_nx_graph(protein_graph))
        return data
    
    def __call__(self):


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

        protein_graphs = {}

        for protein in tqdm(set(self.df['protein_name'])):
            protein_graphs[protein] = self.protein_to_pyg(protein)

        
        with open(os.path.join(self.out_path, 'protein_graph.pkl'), 'wb') as f:
            pickle.dump(protein_graphs, f)
            
        # Process small molecules
        molecule_graphs = {}

        print("Converting smiles to pyg...")

        for smiles in tqdm(set(self.df['molecule_smiles'])):
            molecule_graphs[smiles] = self.smiles_to_pyg(smiles)

        with open(os.path.join(self.out_path, 'molecule_graph.pkl'), 'wb') as f:
            # Dump the data into the file
            pickle.dump(molecule_graphs, f)

if __name__ == "__main__": 
    path = "data/normalized_data.csv"
    out_path = "data/graphs"
    graph_data(path, out_path)()