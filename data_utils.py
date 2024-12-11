from multiprocessing import Pool
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader as PyGDataLoader

from helper import dotdict
from featurization import smile_to_graph


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


def make_leashbio_fold(num_train=None,num_valid=None):

    #make 5% train-valid split
    all       = 98_415_610
    all_valid = 5_000_000
    all_train = 93_415_610
    
    rng = np.random.RandomState(123)
    index = np.arange(all)
    rng.shuffle(index)
    train_index = index[:all_train]
    valid_index = index[all_train:]
    #train_index = np.sort(train_index)
    #valid_index = np.sort(valid_index)

    #subsample according to input arguments
    if num_train is not None:
        train_index=train_index[:num_train]
    if num_valid is not None:
        valid_index=valid_index[:num_valid]

    #check no overlap
    #print('make_fold() overlap:',set(train_index).intersection(set(valid_index))) #set()
    # print('train_index', len(train_index), train_index[[0,1,-1]]) # print some index for debug
    # print('valid_index', len(valid_index), valid_index[[0,1,-1]]) #
    return train_index, valid_index

def split_train_valid(df):
    #make 5% train-valid split
    all_num   = 98_415_610
    valid_num = 5_000_000
    # train_num = 12_000_000
    # train_num = 24_000_000
    # train_num = 6_000_000
    
    # nonshare_bb1_num = 40   # all=271
    # nonshare_bb2_num = 100  # all=693
    # nonshare_bb3_num = 130  # all=871

    nonshare_bb1_num = 30   # all=271
    nonshare_bb2_num = 80  # all=693
    nonshare_bb3_num = 100  # all=871


    np.random.seed(123)
    # np.random.seed(369)
    nonshare_bb1 = np.random.choice(df['buildingblock1_smiles'].unique(), size=nonshare_bb1_num, replace=False)
    nonshare_bb2 = np.random.choice(df['buildingblock2_smiles'].unique(), size=nonshare_bb2_num, replace=False)
    nonshare_bb3 = np.random.choice(df['buildingblock3_smiles'].unique(), size=nonshare_bb3_num, replace=False)

    df['index'] =df.index

    df_nonshare = df.loc[(df['buildingblock1_smiles'].isin(nonshare_bb1)) & (df['buildingblock2_smiles'].isin(nonshare_bb2)) & (df['buildingblock3_smiles'].isin(nonshare_bb3))]
    df_share = df.loc[~(df['buildingblock1_smiles'].isin(nonshare_bb1)) & ~(df['buildingblock2_smiles'].isin(nonshare_bb2)) & ~(df['buildingblock3_smiles'].isin(nonshare_bb3))]

    del df
    gc.collect()

    df_nonshare = df_nonshare.reset_index(drop=True)
    df_share = df_share.reset_index(drop=True)


    rng = np.random.RandomState(123)
    # rng = np.random.RandomState(2)
    # rng = np.random.RandomState(369)
    index = np.arange(len(df_share))
    rng.shuffle(index)
    va_idx = index[:valid_num]
    # tr_idx = index[valid_num:valid_num+train_num]
    tr_idx = index[valid_num:]

    df_tr = df_share.iloc[tr_idx]
    df_va = df_share.iloc[va_idx]
    df_tr = df_tr.reset_index(drop=True)
    df_va = df_va.reset_index(drop=True)

    del df_share
    gc.collect()
    
    cols = ['binds_BRD4', 'binds_HSA', 'binds_sEH']
    cols += ['index']
    df_tr = df_tr[cols]
    df_va = df_va[cols]
    df_nonshare = df_nonshare[cols]
    # 先に定義した方がメモリ節約できるかも？

    # return df_tr[cols], df_va[cols], df_nonshare[cols]
    return df_tr, df_va, df_nonshare 



# ver1 結局ver2もこっち
def my_collate(graph, index=None, device='cpu'):
    if index is None:
        index = np.arange(len(graph)).tolist()
    batch = dotdict(
        x=[],
        edge_index=[],
        edge_attr=[],
        batch=[],
        # idx=index
    )
    offset = 0
    for b, i in enumerate(index):
        N, edge, node_feature, edge_feature = graph[i]
        batch.x.append(node_feature)   #ver2 80
        batch.edge_attr.append(edge_feature)    #ver2 16
        batch.edge_index.append(edge.astype(int) + offset)
        batch.batch += N * [b]
        offset += N
    batch.x = torch.from_numpy(np.concatenate(batch.x)).to(device)
    batch.edge_attr = torch.from_numpy(np.concatenate(batch.edge_attr)).to(device)
    batch.edge_index = torch.from_numpy(np.concatenate(batch.edge_index).T).to(device)
    batch.batch = torch.LongTensor(batch.batch).to(device)
    return batch

# ver2 -> in=smiles convert to graphs in this function!
# ver2 -> in=smiles convert to graphs in this function!
# def my_collate(smiles, smile_graph_dict, index=None, device='cpu'):
#     if index is None:
#         index = np.arange(len(smiles)).tolist()
#     batch = dotdict(
#         x=[],
#         edge_index=[],
#         edge_attr=[],
#         batch=[],
#         # idx=index
#     )
#     offset = 0
#     for b, i in enumerate(index):
#         smile = smiles[i]
#         graph = smile_graph_dict[smile]
#         N, edge, node_feature, edge_feature = graph
#         batch.x.append(node_feature)
#         batch.edge_attr.append(edge_feature)
#         batch.edge_index.append(edge.astype(int) + offset)
#         batch.batch += N * [b]
#         offset += N
#     batch.x = torch.from_numpy(np.concatenate(batch.x)).to(device)
#     batch.edge_attr = torch.from_numpy(np.concatenate(batch.edge_attr)).to(device)
#     batch.edge_index = torch.from_numpy(np.concatenate(batch.edge_index).T).to(device)
#     batch.batch = torch.LongTensor(batch.batch).to(device)
#     return batch


####### dataloaderのためのcollate_fn->並列化に向けて #########
def graph_collate_fn(batch, num_atoms, edge_indexes, node_features, edge_features, mode='train'):
    if mode=='train':
        # num_atoms, edge_indexes, node_features, edge_features, binds = list(zip(*batch))
        indexes, binds = list(zip(*batch))
    else:
        # num_atoms, edge_indexes, node_features, edge_features = list(zip(*batch))
        indexes, _ = list(zip(*batch))
    
    x = []
    edge_attr = []
    edge_index = []
    b = []
    offset = 0
    
    for i,ind in enumerate(indexes):
        N, edge, node_feature, edge_feature = num_atoms[ind], edge_indexes[ind], node_features[ind], edge_features[ind]
        x.append(node_feature)
        edge_attr.append(edge_feature)
        edge_index.append(edge.astype(int) + offset)
        b += N * [i]
        offset += N

    x = torch.from_numpy(np.concatenate(x))
    edge_attr = torch.from_numpy(np.concatenate(edge_attr))
    edge_index = torch.from_numpy(np.concatenate(edge_index).T)
    b = torch.LongTensor(b)
    
    if mode=='train':
        binds = torch.from_numpy(np.array(binds)).float()
        return x, edge_attr, edge_index, b, binds
    else:
        return x, edge_attr, edge_index, b


class CPIDataset(object):

    def __init__(self, df, proteins: List, mode='train'):
        super().__init__()
        self.df = df

        if mode=='train':
            self.df = self.df.rename(columns={'binds_BRD4': 'BRD4', 'binds_HSA': 'HSA', 'binds_sEH': 'sEH'})
            for protein in proteins:
                self.df[protein] = self.df[protein].astype(int)
        
            self._targets = self.df[proteins]
        
        self.indexes = self.df['index']
        # self._smiles = self.df['molecule_smiles']
        # self._graphs = self.df['graphs']
    
    def __len__(self):
        return len(self.df)
        
    @property
    def smiles(self):
        return self._smiles
    @smiles.setter
    def smiles(self, smiles):
        self._smiles = smiles
    
    # @property
    # def graphs(self):
    #     return self._graphs
    # @graphs.setter
    # def graphs(self, graphs):
    #     self._graphs = graphs
        
    @property
    def targets(self):
        return self._targets
    @targets.setter
    def targets(self, targets):
        self._targets = targets

######## 並列化のためのdataloaderのためのdataset ##########

# class CPIDataset(Dataset):

#     def __init__(self, df, proteins: List, mode='train'):
#         super().__init__()
#         self.df = df
#         self.mode = mode
#         if self.mode=='train':
#             self.df = self.df.rename(columns={'binds_BRD4': 'BRD4', 'binds_HSA': 'HSA', 'binds_sEH': 'sEH'})
#             for protein in proteins:
#                 self.df[protein] = self.df[protein].astype(int)
        
#             self._targets = self.df[proteins]
        
#         # self._smiles = self.df['molecule_smiles']
#         # self.num_atoms = df['num_atoms']
#         # self.edge_index = df['edge_index']
#         # self.node_feature = df['node_feature']
#         # self.edge_feature = df['edge_feature']
#         self.df['index'] = self.df['index'].astype(int)
#         self.indexes = self.df['index']
        
#     # @property
#     # def smiles(self):
#     #     return self._smiles
#     # @smiles.setter
#     # def smiles(self, smiles):
#     #     self._smiles = smiles
    
#     # @property
#     # def graphs(self):
#     #     return self._graphs
#     # @graphs.setter
#     # def graphs(self, graphs):
#     #     self._graphs = graphs
        
#     @property
#     def targets(self):
#         return self._targets
#     @targets.setter
#     def targets(self, targets):
#         self._targets = targets
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, i):
#         if self.mode=='train':
#             # return self.num_atoms.iloc[i], self.edge_index.iloc[i], self.node_feature.iloc[i], self.edge_feature.iloc[i], self.targets.iloc[i]
#             return self.indexes.values[i], self.targets.values[i]
#         else:
#             # return self.num_atoms.iloc[i], self.edge_index.iloc[i], self.node_feature.iloc[i], self.edge_feature.iloc[i]
#             return self.indexes.values[i], None