import sys
from logging import Logger
from typing import Callable, Dict, List, Tuple
import os
import pandas as pd
import pickle
import time
import numpy as np
import gc

import torch

from utils import makedirs, timeit, create_logger
from args import TrainArgs
from data_utils import CPIDataset, make_leashbio_fold, split_train_valid
from constants import TRAIN_LOGGER_NAME, MODEL_FILE_NAME
from train import run_training

# from tape import ProteinBertModel, TAPETokenizer
import warnings
warnings.filterwarnings('ignore')


@timeit(logger_name=TRAIN_LOGGER_NAME)

def main(args: TrainArgs,
         train_func
            ) -> Tuple[float, float]:


    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    # init_seed = args.seed
    # save_dir = args.save_dir
    # args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
    #                                  target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    #args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    #set explicit H option and reaction option
    # set_explicit_h(args.explicit_h)
    # set_reaction(args.reaction, args.reaction_mode)


    #SET TOCKENIZER
    # tokenizer = TAPETokenizer(vocab='unirep')

    # Get data
    debug('Loading data')
    """Load preprocessed data."""
    data_dir = args.data_dir
    df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))

    df['index'] = df.index
    cols = ['binds_BRD4', 'binds_HSA', 'binds_sEH']
    cols += ['index']
    df = df[cols]

    # merge graph features
    # debug('Loading graphs num_atoms file')
    # start = time.time()
    # with open(os.path.join(args.graph_dir, 'exp1_ver2_num_atoms.npy'), mode='rb') as f:
    #     num_atoms = np.load(f, allow_pickle=True)
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to load num_atoms file.')
    
    # debug('Loading graphs edge_index file')
    # start = time.time()
    # with open(os.path.join(args.graph_dir, 'exp1_ver2_edge_index.npy'), mode='rb') as f:
    #     edge_indexes = np.load(f, allow_pickle=True)
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to load edge_index file.')
    
    # debug('Loading graphs node_feature file')
    # start = time.time()
    # with open(os.path.join(args.graph_dir, 'exp1_ver2_node_feature.npy'), mode='rb') as f:
    #     node_features = np.load(f, allow_pickle=True)
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to load node_feature file.')
    
    # debug('Loading graphs edge_feature file')
    # start = time.time()
    # with open(os.path.join(args.graph_dir, 'exp1_ver2_edge_feature.npy'), mode='rb') as f:
    #     edge_features = np.load(f, allow_pickle=True)
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to load edge_feature file.')
    
    # debug('finish loading graph data')
    

    # debug('create new column graph')
    # start = time.time()
    # df['graphs'] = all_graphs
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to create graph column.')
    # debug('finish create new column')
    

    train_index, valid_index = make_leashbio_fold()
    # debug('spliting data')
    # start = time.time()
    # df_tr, df_va, df_nonshare = split_train_valid(df=df)
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f}s to split the data.')
    

    # if args.debug_mode:
    #     df_tr = df_tr[:10000]
    #     df_va = df_va[:5000]
    #     df_nonshare = df_nonshare[:5000]
    

    df_tr = df.iloc[train_index]
    df_tr = df_tr.reset_index(drop=True)
    df_va = df.iloc[valid_index]
    df_va = df_va.reset_index(drop=True)
    del df
    gc.collect()

    cols = ['binds_BRD4', 'binds_HSA', 'binds_sEH']
    cols += ['index']
    df_tr = df_tr[cols]
    df_va = df_va[cols]

    train_data = CPIDataset(df_tr, proteins=args.proteins)
    valid_data = CPIDataset(df_va, proteins=args.proteins)
    # nonshare_data = CPIDataset(df_nonshare, proteins=args.proteins)

    # with open(os.path.join(args.graph_dir, 'train_6m.pkl'), mode='wb') as f:
    #     pickle.dump(df_tr, f)
    del df_tr
    # with open(os.path.join(args.graph_dir, 'valid_500k.pkl'), mode='wb') as f:
        # pickle.dump(df_va, f)
    del df_va
    # with open(os.path.join(args.graph_dir, 'nonshare_40_100_130.pkl'), mode='wb') as f:
    #     pickle.dump(df_nonshare, f)
    # del df_nonshare
    gc.collect()

    # Report results
    info('start training and validation!')

    # run training and validation
    # results = train_func(args, train_data, valid_data, nonshare_data, logger)
    results = train_func(args, train_data, valid_data, logger)
    
    info('finish training and validation!')

    return results


if __name__=="__main__":
    main(args=TrainArgs().parse_args(), train_func=run_training)