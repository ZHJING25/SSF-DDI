import os
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from dataset import load_ddi_dataset
from data_preprocessing import CustomData
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv



class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))

        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,35,2)

    def forward(self,data):
        input_feature,edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature,edge_index)
        return intra_rep

# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim,input_dim),32,2)

    def forward(self,h_data,t_data,b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input,t_input),edge_index)
        h_rep = self.inter((t_input,h_input),edge_index[[1,0]])
        return h_rep,t_rep





class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class Net(torch.nn.Module):
    def __init__(self,num_features,nhid=32,num_classes=2,pooling_ratio=0.5,dropout_ratio=0.5):
        super(Net, self).__init__()
        #self.args = args
        self.num_features = num_features
        self.nhid = nhid
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio

        self.sag_conv1 = GCNConv(self.num_features, self.nhid)
        self.sag_pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.sag_conv2 = GCNConv(self.nhid, self.nhid)
        self.sag_pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.sag_conv3 = GCNConv(self.nhid, self.nhid)
        self.sag_pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)


    def sag_encode(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.sag_conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.sag_pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.sag_conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.sag_pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.sag_conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.sag_pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3
        return x

    def forward(self, data):
        x = self.sag_encode(data)
        return x

if __name__ == "__main__":

    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)
    data_iter = iter(train_loader)
    next_batch = next(data_iter)[0]
    x, edge_index, batch = next_batch.x,next_batch.edge_index, next_batch.batch
    # print(x)
    # print(edge_index)
    # print(batch)

    device = torch.device('cuda:0')
    model = Net(70)
    model.to(device)
    print(model(next_batch.to(device)).shape)
    for data in train_loader:
        head_pairs, tail_pairs, rel, label,head_embedding,tail_embedding = [d.to(device) for d in data]
        pred = model(head_pairs.to(device))
        print(pred.shape)

