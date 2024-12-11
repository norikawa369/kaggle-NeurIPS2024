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
from torch.utils.data import DataLoader
from tqdm import trange

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, average_precision_score
import torch
from torch import nn

from optimizers import RAdam, Lookahead
from data_utils import CPIDataset, graph_collate_fn
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


@timeit(logger_name=TEST_LOGGER_NAME)

def make_submission(args: TrainArgs):

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
    df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))

    # Load/build model
    if args.checkpoint_path is not None:
        debug(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(args.checkpoint_path, logger=logger)
    
    molecule_smiles = df['molecule_smiles'].values
    start = time.time()
    with Pool(processes=args.num_workers) as pool:
        test_graphs = list(tqdm(pool.imap(smile_to_graph, molecule_smiles), total=len(molecule_smiles)))
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to convert smiles->graphs.')

    df['graph'] = test_graphs

    debug(df['graph'][0])

    df['num_atoms'] = df['graph'].map(lambda x: x[0])
    df['edge_index'] = df['graph'].map(lambda x: x[1])
    df['node_feature'] = df['graph'].map(lambda x: x[2])
    df['edge_feature'] = df['graph'].map(lambda x: x[3])

    debug('saving the df with graphs.')
    start = time.time()
    with open(os.path.join(args.save_dir, 'test_df.pkl'), mode='wb') as f:
        pickle.dump(df, f)
    
    num_atoms = df['num_atoms'].to_numpy()
    edge_indexes = df['edge_index'].to_numpy()
    node_features = df['node_feature'].to_numpy()
    edge_features = df['edge_feature'].to_numpy()
    
    df['index'] = df.index
    # with open(os.path.join(save_dir, 'train_smile_graph.pkl'), mode='wb') as f:
    #     pickle.dump(train_smile_graphs,f)
    
    # with open(os.path.join(save_dir, 'valid_smile_graph.pkl'), mode='wb') as f:
    #     pickle.dump(valid_smile_graphs,f)
    end = time.time()
    elapsed_time = end-start
    
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to save graphs by pickle.')
    debug('finish saving df with graphs.')

    test_data = CPIDataset(df, proteins=args.proteins, mode='test')
    dataloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=lambda batch: graph_collate_fn(batch, num_atoms, edge_indexes, node_features, edge_features, mode='test'), shuffle=False,
                                pin_memory=True, num_workers=args.num_workers)
    


    info('start infering !')
    P_interactions = []
    model.eval()
    with torch.no_grad():

        for i, (x, edge_attr, edge_index, b) in enumerate(dataloader):
                
            B = len(x)
            
            x = x.to(args.device)
            edge_attr = edge_attr.to(args.device)
            edge_index = edge_index.to(args.device)
            b = b.to(args.device)
            
            graphs = dotdict(
                x=[],
                edge_attr=[],
                edge_index=[],
                batch=[],
            )
            graphs.x = x
            graphs.edge_attr = edge_attr
            graphs.edge_index = edge_index
            graphs.batch = b

            batch = dotdict(
                graph = graphs,
            )

            
            model.output_type = ['infer']
        
            output = model(batch)

            predicted_interactions = output['bind']
            predicted_interactions = predicted_interactions.to('cpu').data.numpy()

            P_interactions.append(predicted_interactions)
    
    info('finish infering')    
    P_interactions = np.concatenate(P_interactions, axis=0)
    for i, protein in enumerate(args.proteins):
        df[protein] = P_interactions[:, i]
    
    df_BRD4 = df[['molecule_smiles', 'BRD4']].reset_index(drop=True).rename(columns={'BRD4': 'binds'})
    df_BRD4['protein_name'] = 'BRD4'

    df_HSA = df[['molecule_smiles', 'HSA']].reset_index(drop=True).rename(columns={'HSA': 'binds'})
    df_HSA['protein_name'] = 'HSA'

    df_sEH = df[['molecule_smiles', 'sEH']].reset_index(drop=True).rename(columns={'sEH': 'binds'})
    df_sEH['protein_name'] = 'sEH'

    sub_df = pd.concat([df_BRD4, df_HSA, df_sEH], ignore_index=True)
    test_df = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/test.csv')

    sub_df = test_df.merge(sub_df, how='left', on=['molecule_smiles', 'protein_name'])
    sub_df = sub_df[['id', 'binds']]
    
    debug(f'sub_df has {len(sub_df)} rows.')
    debug(f'sub_df has {sub_df.isnull().sum()} nulls.')
    
    sample_sub = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/sample_submission.csv')
    debug(f'sample_sub has {len(sample_sub)} rows.')

    return sub_df, args.save_dir

if __name__ == '__main__':
    sub_df, save_dir = make_submission(args=TrainArgs().parse_args())
    # sub_df.to_csv(os.path.join(save_dir, 'submission_6m.csv'), index=False)
    sub_df.to_csv(os.path.join(save_dir, 'submission_ft1.csv'), index=False)

