import numpy as np

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

from args import TrainArgs
from helper import F_unpackbits


PACK_NODE_DIM=9
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8

class MPNNLayer(MessagePassing):
	def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
		super().__init__(aggr=aggr)

		self.emb_dim = emb_dim
		self.edge_dim = edge_dim
		
        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
		self.mlp_msg = nn.Sequential(
			nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)
		
		# MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
		self.mlp_upd = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)

	def forward(self, h, edge_index, edge_attr):
		"""
        The forward pass updates node features `h` via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the 
        message passing procedure: `message()` -> `aggregate()` -> `update()`.
        
        The MessagePassing class handles most of the logic for the implementation.
        To build custom GNNs, we only need to define our own `message()`, 
        `aggregate()`, and `update()` functions (defined subsequently).

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
		out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
		return out

	def message(self, h_i, h_j, edge_attr):
		"""Step (1) Message

        The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take 
        any arguments that were initially passed to `propagate`. Additionally, 
        we can differentiate destination nodes and source nodes by appending 
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`. 
        
        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
		msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
		return self.mlp_msg(msg)

	def aggregate(self, inputs, index):
		"""Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
		return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

	def update(self, aggr_out, h):
		"""
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
		upd_out = torch.cat([h, aggr_out], dim=-1)
		return self.mlp_upd(upd_out)

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModel(nn.Module):
	def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
		super().__init__()

		self.lin_in = nn.Linear(in_dim, emb_dim)

		# Stack of MPNN layers
		self.convs = torch.nn.ModuleList()
		for layer in range(num_layers):
			self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

		self.pool = global_mean_pool

	def forward(self, data): #PyG.Data - batch of PyG graphs

		h = self.lin_in(F_unpackbits(data.x,-1).float())  

		for conv in self.convs:
			h = h + conv(h, data.edge_index.long(), F_unpackbits(data.edge_attr,-1).float())  # (n, d) -> (n, d)

		h_graph = self.pool(h, data.batch)  
		return h_graph

# our prediction model here !!!!
class CPIModel(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.output_type = ['infer', 'loss']
		self.args = args

		self.smile_encoder = MPNNModel(
			 in_dim=self.args.node_dim, edge_dim=self.args.edge_dim, emb_dim=self.args.graph_dims, num_layers=self.args.num_layers,
		)
		self.bind = nn.Sequential(
			nn.Linear(self.args.graph_dims, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 512),
			#nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(512, 3),
		)

	def forward(self, batch):
		graph = batch['graph']
		x = self.smile_encoder(graph) 
		bind = self.bind(x)

		# --------------------------
		output = {}
		if 'loss' in self.output_type:
			target = batch['bind']
			output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())
		if 'infer' in self.output_type:
			output['bind'] = torch.sigmoid(bind)

		return output
	

class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class CrossAttentionBlock(nn.Module):
    """
        The main idea of Perceiver CPI (cross attention block + self attention block).
    """

    def __init__(self, args: TrainArgs):

        super(CrossAttentionBlock, self).__init__()
        self.att_fp_gr = AttentionBlock(hid_dim = 512, n_heads = 8, dropout=args.dropout)
        self.att_gr_gr = AttentionBlock(hid_dim = 512, n_heads = 8, dropout=args.dropout)

    def forward(self,graph_feature,fp_feature):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
        """
        # cross attention for compound information enrichment
        graph_feature = graph_feature + self.att_fp_gr(fp_feature,graph_feature,graph_feature)
        # self-attention
        graph_feature = self.att_gr_gr(graph_feature,graph_feature,graph_feature)
        
        return graph_feature
	
class CPIModel_with_FP(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.output_type = ['infer', 'loss']
		self.args = args

		self.smile_encoder = MPNNModel(
			 in_dim=self.args.node_dim, edge_dim=self.args.edge_dim, emb_dim=self.args.graph_dims, num_layers=self.args.num_layers,
		)
		self.fp_layer = nn.Sequential(
			nn.Linear(2048, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1)
		)
		self.attention = CrossAttentionBlock(self.args)

		self.graph_layer = nn.Sequential(
			nn.Linear(self.args.graph_dims, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 512),
			#nn.BatchNorm1d(512),
			# nn.ReLU(inplace=True),
			# nn.Dropout(0.1),
			# nn.Linear(512, 3),
		)
		self.bind = nn.Linear(512, 3)

	def forward(self, batch):
		graph = batch['graph']
		graph = self.smile_encoder(graph) 
		graph = self.graph_layer(graph)

		fp = batch['fp']
		fp = self.fp_layer(fp)

		bind = self.attention(graph, fp)
		bind = self.bind(bind)

		# --------------------------
		output = {}
		if 'loss' in self.output_type:
			target = batch['bind']
			output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())
		if 'infer' in self.output_type:
			output['bind'] = torch.sigmoid(bind)

		return output
	