import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.
    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.
    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def param_count_all(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    Supports:
    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`
    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)