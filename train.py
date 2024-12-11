import math
import numpy as np
from typing import Any, Callable, List, Tuple, Union, Dict
from logging import Logger
import os
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib import tenumerate
import pickle
import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader as PyGDataLoader

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, average_precision_score
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from optimizers import RAdam, Lookahead
from data_utils import CPIDataset, graph_collate_fn, my_collate
from args import TrainArgs
from utils import makedirs, save_checkpoint, load_checkpoint
from model import CPIModel, CPIModel_with_FP
from nn_utils import param_count, param_count_all
from constants import MODEL_FILE_NAME
from featurization import to_pyg_list, smile_to_graph
from helper import dotdict

# check memory usage
def print_allocated_memory():
    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))

class Trainer(object):
    # def __init__(self, model, lr, weight_decay):
    # def __init__(self, model, num_atoms, edge_indexes, node_features, edge_features, args):
    def __init__(self, model, args):
    # def __init__(self, model, graphs, fps, args):
        self.args = args
        self.model = model
        # self.num_atoms = num_atoms
        # self.edge_indexes = edge_indexes
        # self.node_features = node_features
        # self.edge_features = edge_features
        # self.graphs = graphs
        # self.fps = fps

        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        # loss_func定義!
        # self.loss_func = nn.MSELoss(reduction='mean')
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

        for p in self.model.parameters():
            # if p.dim() > 1:
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        # self.optimizer_inner = RAdam(
        #     [{'params': weight_p, 'weight_decay': args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=args.init_lr)
        # self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.optimizer = RAdam(
            [{'params': weight_p, 'weight_decay': args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=args.init_lr)
        
    # def train(self, train_binds, train_idx):
    def train(self, train_idx, graphs, targets):
    # ver2 -> input smiles convert to graphs in function
    # def train(self, smiles, smile_graph_dict, train_binds):
        self.model.train()
        # dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                # pin_memory=True, num_workers=self.args.num_workers, drop_last=True)
        # np.random.shuffle(dataset)
        n_iter = 0
        
        # use AMP
        scaler = torch.cuda.amp.GradScaler()

        # train_idx = np.arange(len(train_binds))
        # shuffled_idx = train_idx.copy()
        # np.random.shuffle(shuffled_idx)
        # when making fold, idx is shuffled.
        for i, index in enumerate(np.arange(0,len(train_idx),self.args.batch_size)):
        # for i, (indexes, binds) in enumerate(dataloader):
            
            indexes = train_idx[index:index+self.args.batch_size]
            B = len(indexes)
            if B!=self.args.batch_size: continue #drop last

            batch = dotdict(
                graph = my_collate(graphs, indexes, device=self.args.device),
                bind = torch.from_numpy(targets.iloc[index:index+B].values).float().to(self.args.device),
            )


            self.optimizer.zero_grad()
        
            # use AMP
            
            self.model.output_type = ['loss', 'infer']
            with torch.cuda.amp.autocast():

                output = self.model(batch)
                # print("memory after model forward")
                # print_allocated_memory()
                bce_loss = output['bce_loss']
                loss_value = bce_loss.item()
                
                # gradient accumulation
                bce_loss = bce_loss / self.args.iters_accumulate

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(bce_loss).backward()

            # print("memory after backward")
            # print_allocated_memory()
            
            # del 計算グラフによって，メモリを使えるようにする．
            del bce_loss
            torch.cuda.memory_reserved(device=self.args.device)

            # print("memory after clear cache")
            # print_allocated_memory()

            # backwrd only loop per iters_accumulate
            if (i+1) % self.args.iters_accumulate == 0:
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                scaler.update()
            
            torch.clear_autocast_cache()
            n_iter += B
        # return loss_total
        return loss_value, n_iter
    
    # def train(self, dataset):
    #     self.model.train()

    #     dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=lambda batch: graph_collate_fn(batch, self.num_atoms, self.edge_indexes, self.node_features, self.edge_features, mode='train'), shuffle=True,
    #                             pin_memory=True, num_workers=self.args.num_workers, drop_last=True)
    #     # np.random.shuffle(dataset)
    #     n_iter = 0
        
    #     # use AMP
    #     scaler = torch.cuda.amp.GradScaler()

    #     for i, (x, edge_attr, edge_index, b, binds) in enumerate(dataloader):
    #         x = x.to(self.args.device)
    #         edge_attr = edge_attr.to(self.args.device)
    #         edge_index = edge_index.to(self.args.device)
    #         b = b.to(self.args.device)
    #         binds = binds.to(self.args.device)

    #         graphs = dotdict(
    #             x=[],
    #             edge_attr=[],
    #             edge_index=[],
    #             batch=[],
    #         )
    #         graphs.x = x
    #         graphs.edge_attr = edge_attr
    #         graphs.edge_index = edge_index
    #         graphs.batch = b

    #         batch = dotdict(
    #             graph = graphs,
    #             bind = binds,
    #         )

    #         self.optimizer.zero_grad()
        
    #         # use AMP
            
    #         self.model.output_type = ['loss', 'infer']
    #         with torch.cuda.amp.autocast():

    #             output = self.model(batch)
    #             # print("memory after model forward")
    #             # print_allocated_memory()
    #             bce_loss = output['bce_loss']
    #             loss_value = bce_loss.item()
                
    #             # gradient accumulation
    #             bce_loss = bce_loss / self.args.iters_accumulate

    #         # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
    #         scaler.scale(bce_loss).backward()

    #         # print("memory after backward")
    #         # print_allocated_memory()
            
    #         # del 計算グラフによって，メモリを使えるようにする．
    #         del bce_loss
    #         torch.cuda.memory_reserved(device=self.args.device)

    #         # print("memory after clear cache")
    #         # print_allocated_memory()

    #         # backwrd only loop per iters_accumulate
    #         if (i+1) % self.args.iters_accumulate == 0:
    #             # scaler.step() first unscales the gradients of the optimizer's assigned params.
    #             scaler.step(self.optimizer)

    #             # Updates the scale for next iteration.
    #             scaler.update()
            
    #         torch.clear_autocast_cache()
    #         n_iter += self.args.batch_size
    #     # return loss_total
    #     return loss_value, n_iter
    
    
    # def eval(self, valid_binds, valid_idx):
    # ver2 -> input smiles
    def eval(self, valid_idx, graphs, targets):
        self.model.eval()

        # valid_graphs = to_pyg_list(valid_graphs)
        # dataloader = PyGDataLoader(valid_graphs, batch_size=self.args.batch_size, shuffle=False)
        # dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                # pin_memory=True, num_workers=self.args.num_workers)
        
        T_Interactions, P_Interactions = [], []
        with torch.no_grad():

            # for i, (indexes, binds) in enumerate(dataloader):
            
            for i, index in enumerate(np.arange(0,len(valid_idx),self.args.batch_size)):
                
                indexes = valid_idx[index:index+self.args.batch_size]
                
                B = len(indexes)
                # not drop last
                batch = dotdict(
                    graph = my_collate(graphs,indexes,device=self.args.device),
                    bind = torch.from_numpy(targets.iloc[index:index+B].values).float().to(self.args.device),
                )


                # B = len(binds)
                
                # x = x.to(self.args.device)
                # edge_attr = edge_attr.to(self.args.device)
                # edge_index = edge_index.to(self.args.device)
                # b = b.to(self.args.device)
                # binds = binds.to(self.args.device)

                # graphs = dotdict(
                #     x=[],
                #     edge_attr=[],
                #     edge_index=[],
                #     batch=[],
                # )
                # graphs.x = x
                # graphs.edge_attr = edge_attr
                # graphs.edge_index = edge_index
                # graphs.batch = b

                # batch = dotdict(
                #     graph = graphs,
                #     bind = binds,
                # )


                self.model.output_type = ['loss', 'infer']
            
                output = self.model(batch)
                
                val_loss = output['bce_loss']
                val_loss = val_loss.item()
                
                binds = batch['bind'].to('cpu').data.numpy()
                
                predicted_interactions = output['bind']
                predicted_interactions = predicted_interactions.to('cpu').data.numpy()
                
                T_Interactions.append(binds)
                P_Interactions.append(predicted_interactions)
        # AUC = roc_auc_score(T, S)
        # tpr, fpr, _ = precision_recall_curve(T, S)
        # PRC = auc(fpr, tpr)
        T_Interactions = np.concatenate(T_Interactions, axis=0)
        P_Interactions = np.concatenate(P_Interactions, axis=0)
        APS = average_precision_score(T_Interactions, P_Interactions, average='micro')

        APS_proteins = dotdict(protein=self.args.proteins, score=[])

        for i, protein in enumerate(self.args.proteins):
            aps = average_precision_score(T_Interactions[:, i], P_Interactions[:, i])
            APS_proteins.score.append(aps)
            
        AUC = roc_auc_score(T_Interactions, P_Interactions, average='micro')
        # tpr, fpr, _ = precision_recall_curve(T_Interactions, P_Interactions)
        # PRC = auc(fpr, tpr)

        return val_loss, APS, APS_proteins, AUC, T_Interactions, P_Interactions

'''def run_training(args: TrainArgs, train_data, validation_data, logger: Logger):

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # torch.backends.cudnn.deterministic = True
    
    """Create a dataset and split it into train/dev/test."""
    train_data_size = len(train_data.smiles)
    valid_data_size = len(validation_data.smiles)
    
    debug(f'train size = {train_data_size:,} | val size = {valid_data_size:,}')

    # debug(f'train pos/neg = {tr_pos_num}/{tr_neg_num} | val pos/neg = {va_pos_num}/{va_neg_num}')


    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    """Start training."""
    
    save_dir = os.path.join(args.save_dir)
    makedirs(save_dir)

    writer = SummaryWriter(log_dir=save_dir)

    # Load/build model
    if args.checkpoint_paths is not None:
        debug(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(args.checkpoint_path, logger=logger)
    else:
        debug(f'Building model')
        
        model = CPIModel(args)

    debug(model)

    if args.checkpoint_frzn is not None:
        debug(f'Number of unfrozen parameters = {param_count(model):,}')
        debug(f'Total number of parameters = {param_count_all(model):,}')
    else:
        debug(f'Number of parameters = {param_count_all(model):,}')

    if args.cuda:
        debug('Moving model to cuda')
    model = model.to(args.device)

    # Ensure that model is saved in correct location for evaluation if 0 epochs
    save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, args)

    # move eval funtion to trainer
    # trainer = Trainer(model, lr, weight_decay, batch)
    trainer = Trainer(model, args)

    # tester = Tester(model, device)
    # tester = Tester(model, args)

    # Run training
    # start = time.time()
    # with Pool(processes=36) as pool:
    #     train_graphs = list(tqdm(pool.imap(smile_to_graph, train_data.smiles), total=train_data_size))
    
    # with Pool(processes=36) as pool:
    #     valid_graphs = list(tqdm(pool.imap(smile_to_graph, validation_data.smiles), total=valid_data_size))
    # end = time.time()
    # elapsed_time = end-start
    # debug(f'{elapsed_time // 3600}:{(elapsed_time % 3600) // 60}:{(elapsed_time % 3600 % 60)} to convert smiles->graphs.')
    
    # with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp1/output/ver2', 'train_smile_graph.pkl'), mode='rb') as f:
    #     train_graphs = pickle.load(f)
    # with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp1/output/ver2', 'valid_smile_graph.pkl'), mode='rb') as f:
    #     valid_graphs = pickle.load(f) 
    
    start = time.time()
    with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp1/output/ver2', 'all_smiles_graphs.pkl'), mode='rb') as f:
        all_smiles_graphs = pickle.load(f) 
    end = time.time()
    elapsed_time = end-start
    debug(f'{elapsed_time // 3600}:{(elapsed_time % 3600) // 60}:{(elapsed_time % 3600 % 60)} to load smiles_graphs_dictionary')
    
    # train_graphs = list(train_graphs.values())
    # valid_graphs = list(valid_graphs.values())

    # train_smile_graphs = dict(zip(train_data.smiles.values, train_graphs))
    # valid_smile_graphs = dict(zip(validation_data.smiles.values, valid_graphs))
    
    ######################### graphs->smilesに修正####################################
    best_loss = np.inf
    best_epoch, n_iter_sum = 0, 0

    # for epoch in range(1, iteration+1):
    for epoch in trange(1, args.epochs+1):
        debug(f'Epoch {epoch}')

        if epoch % args.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= args.lr_decay

        loss_train, n_iter = trainer.train(train_smiles=train_data.smiles, smile_graph_dict=all_smiles_graphs, train_binds=train_data.targets)
        loss_val, APC, AUC, T_Interactions, P_Interactions = trainer.eval(valid_smiles=validation_data.smiles, smile_graph_dict=all_smiles_graphs, valid_binds=validation_data.targets)
        
        n_iter_sum += n_iter

        debug(f'Train Loss = {loss_train:.6f}')
        writer.add_scalar(f'train_loss', loss_train, n_iter_sum)

        debug(f'Validation Loss = {loss_val:.6f}')
        writer.add_scalar(f'validation_loss', loss_val, n_iter_sum)
        debug(f'Validation APC = {APC:.6f}')
        writer.add_scalar(f'Validation_APC', APC, n_iter_sum)
        debug(f'Validation AUC = {AUC:.6f}')
        writer.add_scalar(f'Validation_AUC', AUC, n_iter_sum)
        # debug(f'Validation PRC = {PRC:.6f}')
        # writer.add_scalar(f'Validation_PRC', PRC, n_iter_sum)
        # Save model checkpoint if improved validation score
        if loss_val < best_loss:
            best_loss, best_epoch = loss_val, epoch

            best_APC = APC
            best_AUC = AUC
            # best_PRC = PRC

            debug(f'Save the model on epoch {best_epoch}!')
            save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, args)

    info(f'Model best validation BCE = {best_loss:.6f} on epoch {best_epoch}')
    info(f'validation APC = {best_APC:.6f} on epoch {best_epoch}')
    # info(f'validation AUC = {best_AUC:.6f}, PRC = {best_PRC: .6f} on epoch {best_epoch}')
    info(f'validation AUC = {best_AUC:.6f} on epoch {best_epoch}')


    
    results = dict()
    results['loss'] = best_loss
    results['APC'] = best_APC
    results['AUC'] = best_AUC
    # results['PRC'] = best_PRC
    
    return results'''

# def run_training(args: TrainArgs, train_data: CPIDataset, validation_data: CPIDataset, nonshare_data:CPIDataset,  logger: Logger):
def run_training(args: TrainArgs, train_data: CPIDataset, validation_data: CPIDataset, logger: Logger):

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # torch.backends.cudnn.deterministic = True
    
    """Create a dataset and split it into train/dev/test."""
    train_data_size = len(train_data)
    valid_data_size = len(validation_data)
    # nonshare_data_size = len(nonshare_data)
    
    # debug(f'train size = {train_data_size:,} | val size = {valid_data_size:,} | nonshare size = {nonshare_data_size:,}')
    debug(f'train size = {train_data_size:,} | val size = {valid_data_size:,}')

    # debug(f'train pos/neg = {tr_pos_num}/{tr_neg_num} | val pos/neg = {va_pos_num}/{va_neg_num}')


    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    """Start training."""
    save_dir = os.path.join(args.save_dir)
    # save_dir = os.path.join(args.save_dir, 'fine_tune1')
    makedirs(save_dir)

    writer = SummaryWriter(log_dir=save_dir)

    # Load/build model
    if args.checkpoint_path is not None:
        debug(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(args.checkpoint_path, logger=logger)
    else:
        debug(f'Building model')
        
        model = CPIModel(args)

    debug(model)

    if args.checkpoint_frzn is not None:
        debug(f'Number of unfrozen parameters = {param_count(model):,}')
        debug(f'Total number of parameters = {param_count_all(model):,}')
    else:
        debug(f'Number of parameters = {param_count_all(model):,}')

    if args.cuda:
        debug('Moving model to cuda')
    model = model.to(args.device)

    # Ensure that model is saved in correct location for evaluation if 0 epochs
    save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, args)

    debug('loading smiles-graphs dictionary.')
    start = time.time()
    # with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/custom_data/smiles-graphs-dict/', 'exp1_ver1_all_smiles_graphs.pkl'), mode='rb') as f:
    #     all_graphs = pickle.load(f) 
    with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/custom_data/', 'exp1_ver2_all_graphs.pkl'), mode='rb') as f:
        all_graphs = pickle.load(f) 
    
    end = time.time()
    elapsed_time = end-start
    debug(f'{(elapsed_time // 3600):.0f}h{((elapsed_time % 3600) // 60):.0f}m{(elapsed_time % 3600 % 60):.0f} to load smiles_graphs_dictionary')
    
    # all_graphs = list(all_graphs.values())
    all_graphs = list(all_graphs)
    # train_graphs = list(train_graphs.values())
    # valid_graphs = list(valid_graphs.values())


    # move eval funtion to trainer
    # trainer = Trainer(model, lr, weight_decay, batch)
    trainer = Trainer(model, args)

    best_loss = np.inf
    best_APS = 0
    best_epoch, n_iter_sum = 0, 0

    # for epoch in range(1, iteration+1):
    for epoch in trange(1, args.epochs+1):
        debug(f'Epoch {epoch}')

        if epoch % args.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= args.lr_decay

        # loss_train, n_iter = trainer.train(train_data.targets, train_data.indexes)
        loss_train, n_iter = trainer.train(train_idx=train_data.indexes, graphs=all_graphs, targets=train_data.targets)
        # loss_val, APS, APS_proteins, AUC, T_Interactions, P_Interactions = trainer.eval(validation_data.targets, validation_data.indexes)
        loss_val, APS, APS_proteins, AUC, T_Interactions, P_Interactions = trainer.eval(valid_idx=validation_data.indexes, graphs=all_graphs, targets=validation_data.targets)
        # ns_loss_val, ns_APS, ns_APS_proteins, ns_AUC, ns_T_Interactions, ns_P_Interactions = trainer.eval(nonshare_data.targets, nonshare_data.indexes)
        
        n_iter_sum += n_iter


        debug(f'Train Loss = {loss_train:.6f}')
        writer.add_scalar(f'train_loss', loss_train, n_iter_sum)

        # validation
        debug(f'Validation Loss = {loss_val:.6f}')
        writer.add_scalar(f'validation_loss', loss_val, n_iter_sum)
        debug(f'Validation APS = {APS:.6f}')
        writer.add_scalar(f'validation_APS', APS, n_iter_sum)

        for i,protein in enumerate(APS_proteins.protein):
            debug(f'Validation APS of {protein} = {APS_proteins.score[i]:.6f}')
            writer.add_scalar(f'validation_APS_{protein}', APS_proteins.score[i], n_iter_sum)

        debug(f'Validation AUC = {AUC:.6f}')
        writer.add_scalar(f'validation_AUC', AUC, n_iter_sum)
        
        # valid of nonshare
        # debug(f'Nonshare Validation Loss = {ns_loss_val:.6f}')
        # writer.add_scalar(f'nonshare_validation_loss', ns_loss_val, n_iter_sum)
        # debug(f'Nonshare Validation APS = {ns_APS:.6f}')
        # writer.add_scalar(f'nonshare_validation_APS', ns_APS, n_iter_sum)

        # for i,protein in enumerate(ns_APS_proteins.protein):
        #     debug(f'Nonshare Validation APS of {protein} = {ns_APS_proteins.score[i]:.6f}')
        #     writer.add_scalar(f'nonshare_validation_APS_{protein}', ns_APS_proteins.score[i], n_iter_sum)

        # debug(f'Nonshare Validation AUC = {ns_AUC:.6f}')
        # writer.add_scalar(f'nonshare_validation_AUC', ns_AUC, n_iter_sum)
        
        # Save model checkpoint if improved validation score
        # if loss_val < best_loss:
        if APS > best_APS:
            best_loss, best_epoch = loss_val, epoch

            best_APS = APS
            best_AUC = AUC
            best_APS_proteins = APS_proteins

            # best_ns_loss = ns_loss_val
            # best_ns_APS = ns_APS
            # best_ns_AUC = ns_AUC
            # best_ns_APS_proteins = ns_APS_proteins
            
            debug(f'Save the model on epoch {best_epoch}!')
            save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, args)

    # info(f'Model best validation BCE = {best_loss:.6f} on epoch {best_epoch}')
    info(f'Model best APS = {best_APS:.6f} on epoch {best_epoch}')
    # info(f'validation APS = {best_APS:.6f} on epoch {best_epoch}')
    info(f'validation BCE = {best_loss:.6f} on epoch {best_epoch}')
    info(f'validation AUC = {best_AUC:.6f} on epoch {best_epoch}')
    for i,protein in enumerate(best_APS_proteins.protein):
        info(f'validation APS of {protein} = {best_APS_proteins.score[i]:.6f} on epoch {best_epoch}')

    # info(f'Model best nonshare validation BCE = {best_ns_loss:.6f} on epoch {best_epoch}')
    # info(f'nonshare validation APS = {best_ns_APS:.6f} on epoch {best_epoch}')
    # info(f'nonshare validation AUC = {best_ns_AUC:.6f} on epoch {best_epoch}')
    # for i,protein in enumerate(best_ns_APS_proteins.protein):
    #     info(f'nonshare validation APS of {protein} = {best_ns_APS_proteins.score[i]:.6f} on epoch {best_epoch}')
        
    
    results = dict()
    results['loss'] = best_loss
    results['APS'] = best_APS
    results['AUC'] = best_AUC
    results['APS_proteins'] = best_APS_proteins
    # results['ns_loss'] = best_ns_loss
    # results['ns_APS'] = best_ns_APS
    # results['ns_AUC'] = best_ns_AUC
    # results['ns_APS_proteins'] = best_ns_APS_proteins
    
    return results