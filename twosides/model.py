from audioop import bias
from bisect import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  global_add_pool, global_mean_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import SAGPool
from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    LayerNorm,
    global_add_pool,
    Set2Set,
)
from layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
)

class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)
        self.sag_pooling = SAGPooling(64, min_score=-1)

    def forward(self, x, edge_index, batch):
        # x_conv = self.conv(x, edge_index)
        # scores = softmax(x_conv, batch, dim=0)
        # gx = global_add_pool(x * scores, batch)
        # readout

        att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.sag_pooling(x, edge_index, batch=batch)
        global_graph_emb = global_add_pool(att_x, h_att_batch)

        return global_graph_emb


class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):

        edge_index = data.edge_index

        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x
    
class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x

class SF_DDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super(SF_DDI, self).__init__()

        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.rmodule = nn.Embedding(963, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)


        #cnn
        self.drug_MAX_LENGH = 100
        self.drug_vocab_size = 65
        self.dim = 64
        self.conv = 40
        self.drug_kernel = [4, 6, 8]
        self.durg_dim_afterCNNs = self.drug_MAX_LENGH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.attention_dim = 40 * 4
        self.mix_attention_head = 5

        self.drug_embed1 = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.drug_embed2 = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)
        self.Drug_max_pool = nn.MaxPool1d(self.durg_dim_afterCNNs)
        self.dropout1 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(320, 128)
        #cnn
        self.cls = nn.Linear(hidden_dim,2)
        ##GAT
        self.initial_norm = LayerNorm(54)
        self.gat_conv = GATConv(54,32,2)
        self.intra_attn = IntraGraphAttention(64)

    def forward(self, triples,head_embedding,tail_embedding):
        h_data, t_data, rels = triples

        # print(head_embedding.shape)
        # print(tail_embedding.shape)

        heads_embed = self.drug_embed1(head_embedding) #[512,100,64]
        tails_embed = self.drug_embed2(tail_embedding) #[512,100,64]
        heads_embed = heads_embed.permute(0, 2, 1) #[512,64,100]
        tails_embed = tails_embed.permute(0, 2, 1) #[512,64,100]
        headsConv = self.Drug_CNNs(heads_embed) #[512,160,85]
        tailsConv = self.Drug_CNNs(tails_embed) #[512,160,85]

        heads_QKV = headsConv.permute(2, 0, 1)
        tails_QKV = tailsConv.permute(2, 0, 1)

        heads_att, _ = self.mix_attention_layer(heads_QKV, tails_QKV, tails_QKV)
        tails_att, _ = self.mix_attention_layer(tails_QKV, heads_QKV, heads_QKV)

        heads_att = heads_att.permute(1, 2, 0)
        tails_att = tails_att.permute(1, 2, 0)

        headsConv = headsConv * 0.5 + heads_att * 0.5
        tailsConv = tailsConv * 0.5 + tails_att * 0.5

        headsConv = self.Drug_max_pool(headsConv).squeeze(2)
        tailsConv = self.Drug_max_pool(tailsConv).squeeze(2)
        pair_conv = torch.cat([headsConv, tailsConv], dim=1)
        pair_conv = self.dropout1(pair_conv)
        pair_conv = self.leaky_relu(self.fc1(pair_conv))
        # print(pair_conv.shape)

        #print(h_data.x.shape,t_data.x.shape)
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        h_data.x = self.gat_conv(h_data.x, h_data.edge_index)
        t_data.x = self.gat_conv(t_data.x, t_data.edge_index)

        #print(h_data.x.shape,t_data.x.shape)
        h_data.x = self.intra_attn(h_data)
        t_data.x = self.intra_attn(t_data)
        # print(self.intra_attn(h_data).shape)
        # print(self.intra_attn(t_data).shape)

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)
        # Start of SSIM
        # TAGP, Equation (8)
        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h_align = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        h_final = global_add_pool(x_h * g_t_align , h_data.batch)
        t_final = global_add_pool(x_t * g_h_align , t_data.batch)


        pair = torch.cat([h_final, t_final], dim=-1)
        #print(pair.shape)
        #print(pair_conv.shape)
        pair = torch.cat([pair_conv,pair], dim=-1)
        # print(pair.shape,pair_conv.shape,pair_sag.shape)
        #print(pair.shape)

        rfeat = self.rmodule(rels)
        #print(rfeat.shape)
        logit = (self.lin(pair) * rfeat).sum(-1)
        #print(logit.shape)
        return logit
