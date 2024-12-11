import sys
import math
import numpy as np
from typing import Any, Callable, List, Tuple, Union, Dict
from logging import Logger
import os
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import trange
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, average_precision_score
import torch
from torch import nn

from optimizers import RAdam, Lookahead
from data_utils import CPIDataset, my_collate
from args import TrainArgs
from utils import makedirs, save_checkpoint, load_checkpoint, timeit, create_logger
from model import CPIModel
from nn_utils import param_count, param_count_all
from constants import MODEL_FILE_NAME
from featurization import to_pyg_list, smile_to_graph
from helper import dotdict
from constants import TRAIN_LOGGER_NAME, MODEL_FILE_NAME, TEST_LOGGER_NAME

# from tape import ProteinBertModel, TAPETokenizer
import warnings
warnings.filterwarnings('ignore')


@timeit(logger_name=TRAIN_LOGGER_NAME)

def make_save_graphs(args: TrainArgs):

    logger = create_logger(name=TEST_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)

    # Get data
    debug('Loading data')
    """Load preprocessed data."""
    # data_dir = '/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/'
    data_dir = args.data_dir
    df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))

    # train_data = CPIDataset(df, proteins=args.proteins, mode='train')
    
    molecule_smiles = df['molecule_smiles'].values
    start = time.time()
    with Pool(processes=args.num_workers) as pool:
        train_graphs = list(tqdm(pool.imap(smile_to_graph, molecule_smiles), total=len(molecule_smiles)))
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to convert smiles->graphs.')

    # train_smiles_graphs = dict(zip(train_data.smiles.values, train_graphs))
    df['graph'] = train_graphs

    debug(df['graph'][0])

    df['num_atoms'] = df['graph'].map(lambda x: x[0])
    df['edge_index'] = df['graph'].map(lambda x: x[1])
    df['node_feature'] = df['graph'].map(lambda x: x[2])
    df['edge_feature'] = df['graph'].map(lambda x: x[3])


    debug('start saving column of graph')
    start = time.time()
    data_dir = os.path.join('/home/akawa005/kaggle-comps/kaggle-leashbio-belka/input', 'custom_data')
    with open(os.path.join(data_dir, 'exp1_ver2_num_atoms.npy'), mode='wb') as f:
        np.save(f, df['num_atoms'].to_numpy())
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save num_atoms.')
    
    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_edge_index.npy'), mode='wb') as f:
        np.save(f, df['edge_index'].to_numpy())
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save edge index.')
    
    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_node_feature.npy'), mode='wb') as f:
        np.save(f, df['node_feature'].to_numpy())
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save node feature.')

    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_edge_feature.npy'), mode='wb') as f:
        np.save(f, df['edge_feature'].to_numpy())
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save edge feature.')
    
    
    debug('check data')
    debug(f'{df.isnull().sum()}')
    return None


def save_df_with_graph(args):
    logger = create_logger(name=TEST_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug('loading train.parquet')
    df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    
    debug('loading smiles graphs dict')
    start = time.time()
    with open(os.path.join(args.save_dir, 'all_smiles_graphs.pkl'), mode='rb') as f:
        all_smiles_graphs = pickle.load(f)
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to load graphs.')
    
    
    debug('create new column graph')
    start = time.time()
    df['graph'] = df['molecule_smiles'].map(lambda x: all_smiles_graphs[x])
    end = time.time()
    elapsed_time = end-start
    
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to create graph column.')
    debug('finish create new column')
    debug(df['graph'][0])

    df['num_atoms'] = df['graph'].map(lambda x: x[0])
    df['edge_index'] = df['graph'].map(lambda x: x[1])
    df['node_feature'] = df['graph'].map(lambda x: x[2])
    df['edge_feature'] = df['graph'].map(lambda x: x[3])


    debug('start saving column of graph')
    start = time.time()
    data_dir = os.path.join('/home/akawa005/kaggle-comps/kaggle-leashbio-belka/input', 'custom_data')
    with open(os.path.join(data_dir, 'exp1_ver2_num_atoms.npy'), mode='wb') as f:
        np.save(f, df['num_atoms'].values)
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save num_atoms.')
    
    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_edge_index.npy'), mode='wb') as f:
        np.save(f, df['edge_index'].values)
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save edge index.')
    
    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_node_feature.npy'), mode='wb') as f:
        np.save(f, df['node_feature'].values)
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save node feature.')

    start = time.time()
    with open(os.path.join(data_dir, 'exp1_ver2_edge_feature.npy'), mode='wb') as f:
        np.save(f, df['edge_feature'].values)
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save edge feature.')
    
    
    debug('check data')
    debug(f'{df.isnull().sum()}')
    return None

if __name__ == '__main__':
    make_save_graphs(args=TrainArgs().parse_args())
    # save_df_with_graph(args=TrainArgs().parse_args())