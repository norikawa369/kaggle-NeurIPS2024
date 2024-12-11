import numpy as np
from tqdm import tqdm

import rdkit
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

import torch
import torch.nn as nn
import torch.nn.functional as F


# mol to graph adopted from
# from https://github.com/LiZhang30/GPCNDTA/blob/main/utils/DrugGraph.py

PACK_NODE_DIM=10 #9->10
PACK_EDGE_DIM=2 #1->2
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))


#Get features of an atom (one-hot encoding:)
'''
	1.atom element: 44+1 dimensions    
	2.the atom's hybridization: 5 dimensions
	3.degree of atom: 6 dimensions                        
	4.total number of H bound to atom: 6 dimensions
	5.number of implicit H bound to atom: 6 dimensions    
	6.whether the atom is on ring: 1 dimension
	7.whether the atom is aromatic: 1 dimension           
	Total: 70 dimensions -> ver2 79(formal charge 5, chiral tag 4)
'''

ATOM_SYMBOL = [
	'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
	'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
	'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
	'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
	'Pt', 'Hg', 'Pb', 'Dy',
	#'Unknown'
]
#print('ATOM_SYMBOL', len(ATOM_SYMBOL))44
HYBRIDIZATION_TYPE = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D
]

# ver2 -> formalcharge,chiraltag足した。70->79
def get_atom_feature(atom):
	# ver2
	feature = (
		 one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL)
	   + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
	   + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
	   + one_of_k_encoding(int(atom.GetChiralTag()), [0, 1, 2, 3])
	   + [atom.IsInRing()]
	   + [atom.GetIsAromatic()]
	)
	# ver1
	# feature = (
	# 	 one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL)
	#    + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
	#    + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	#    + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
	#    + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	#    + [atom.IsInRing()]
	#    + [atom.GetIsAromatic()]
	# )
	# feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


#Get features of an edge (one-hot encoding)
'''
	1.single/double/triple/aromatic: 4 dimensions       
	2.the atom's hybridization: 1 dimensions
	3.whether the bond is on ring: 1 dimension          
	Total: 6 dimensions -> ver2 12(streo 6)
'''

def get_bond_feature(bond):
	bond_type = bond.GetBondType()
	feature = [
		bond_type == Chem.rdchem.BondType.SINGLE,
		bond_type == Chem.rdchem.BondType.DOUBLE,
		bond_type == Chem.rdchem.BondType.TRIPLE,
		bond_type == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	feature += one_of_k_encoding(int(bond.GetStereo()), [0,1,2,3,4,5])
	feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


def smile_to_graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	N = mol.GetNumAtoms()
	node_feature = []
	edge_feature = []
	edge = []
	for i in range(mol.GetNumAtoms()):
		atom_i = mol.GetAtomWithIdx(i)
		atom_i_features = get_atom_feature(atom_i)
		node_feature.append(atom_i_features)

		for j in range(mol.GetNumAtoms()):
			bond_ij = mol.GetBondBetweenAtoms(i, j)
			if bond_ij is not None:
				edge.append([i, j])
				bond_features_ij = get_bond_feature(bond_ij)
				edge_feature.append(bond_features_ij)
	node_feature=np.stack(node_feature)
	edge_feature=np.stack(edge_feature)
	edge = np.array(edge,dtype=np.uint8)
	return N,edge,node_feature,edge_feature

def to_pyg_format(N,edge,node_feature,edge_feature):
	graph = Data(
		idx=-1,
		edge_index = torch.from_numpy(edge.T).int(),
		x          = torch.from_numpy(node_feature).byte(),
		edge_attr  = torch.from_numpy(edge_feature).byte(),
	)
	return graph

def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph
