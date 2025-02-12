import math
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
import torch
from torch import nn
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gcn_lib.torch_nn import BasicConv, batched_index_select, act_layer
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
import json 

class SinusoidalEncoder(nn.Module):

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            'scales', torch.tensor([2**i for i in range(min_deg, max_deg)]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))

class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, k, act='relu', norm=None, bias=True):
        super(GraphAttention, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)
        self.graph_attention = graph_attention(dim=in_channels, embed_dim=in_channels,k=k)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = self.graph_attention(x,x_j)
        temp = x
        x = self.nn(torch.cat([x, x_j], dim=1))
        x = self.relu(x)
        x = x + temp
        return x

class graph_attention(nn.Module):
    def __init__(self, dim, embed_dim,k):

        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.k = k
        self.embedding = nn.Linear(dim,embed_dim)
        self.attention = nn.Linear(2 * embed_dim, 1)
        self.relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, x_n):
        x = x.repeat(1,1,1,self.k)
        x = rearrange(x,'b d x y -> b x y d')
        x_n = rearrange(x_n, 'b d x y -> b x y d')
        e, e_n = self.embedding(x), self.embedding(x_n)
        e = torch.concat((e, e_n), dim=3)
        attn = self.relu(self.attention(e))
        attn = self.softmax(attn)
        x_n = x_n * attn
        x_n = torch.sum(x_n, dim=2)
        x_n = rearrange(x_n, 'b x y -> b y x').unsqueeze(dim=-1)
        return x_n

def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

class knn(torch.nn.Module):
    def __init__(self,k):
        super(knn, self).__init__()
        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        x = F.normalize(x, p=2.0, dim=1)
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=self.k)  #b, n, k
        h,w = 2,1
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index

class knn_euclidean(torch.nn.Module):
    def __init__(self, k, threshold=0.5):
        super(knn_euclidean, self).__init__()
        self.k = k
        self.threshold = threshold
        x = torch.arange(8)
        y = torch.arange(8)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coord = torch.stack([grid_x, grid_y], dim=0).to(torch.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.coord = rearrange(coord, 'd h w -> d (h w)').to(device)

    def forward(self, coords):
        x = coords
        batch_size, n_points, n_dims = x.shape
        # 计算两两之间的距离
        dist = pairwise_distance(x.detach())
        # 将大于阈值的距离设置为无穷大
        dist[dist > self.threshold] = float('inf')
        # 选择最近的 k 个点
        _, nn_idx = torch.topk(-dist, k=self.k)  # b, n, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index

class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()


    def forward(self, feat):
        return feat.mean(0)
        
class Block(torch.nn.Module):

    def __init__(self,in_channels, out_channels, k, euclidean=False,ConvType = 'GraphAttention'):
        super(Block, self).__init__()
        self.k = k
        self.euclidean = euclidean
        if euclidean == False:
            self.knn_graph = knn(k=self.k)
        else:
            self.knn_graph = knn_euclidean(k=self.k)
        if ConvType == 'GraphAttention':
            self.conv = GraphAttention(in_channels, out_channels, k=self.k, act='relu', norm='batch', bias=True)
        elif ConvType == 'GraphSAGE':
            self.conv = GraphSAGE(in_channels, out_channels, act='relu', norm='batch', bias=True)
        elif ConvType == 'EdgeConv2d':
            self.conv = EdgeConv2d(in_channels, out_channels, act='relu', norm='batch', bias=True)
        elif ConvType == 'MRConv2d':
            self.conv = MRConv2d(in_channels, out_channels, act='relu', norm='batch', bias=True)
        # self.down = Downsample(in_channels, out_channels)

    def forward(self, x,coords=None):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        if self.euclidean:
            edge_index = self.knn_graph(coords)
        else:
            edge_index = self.knn_graph(x)
        x = self.conv(x, edge_index)
        x = x.reshape(B, -1, H, W).contiguous()
        # x = self.down(x)
        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)
        # self.label_same_matrix = self.label_same_matrix.to(attn.device)
        # attn = attn * self.label_same_matrix * 2 + attn * (1-self.label_same_matrix)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn
    
class GlobalAttentionBlock(nn.Module):
    def __init__(self,dim,n_head=8) -> None:
        super().__init__()
        
        self.node_norm = nn.LayerNorm(dim)
        self.node_transformer = MultiHeadAttention(n_head, dim, 0.1)
    def forward(self,x,attn_mask=None):
        x = self.node_norm(x)
        x, attn = self.node_transformer(x, x, x, attn_mask)
        return x 


def from_pesudol_label_get_mask(pseudo_label):
    B, L= pseudo_label.shape
    attn_mask = torch.zeros(B, L, L).to(pseudo_label.device)
    for i in range(B):
        attn_mask[i] = (pseudo_label[i] == pseudo_label[i].unsqueeze(1)).float()
    return attn_mask

class Graph_HNet(torch.nn.Module):
    def __init__(self, in_channel,config):
        super(Graph_HNet, self).__init__()
        self.nb_classes = config.get('nb_classes', 5)
        self.add_pesudo_layer = config.get('add_pesudo_layer',True)
        self.channels = config.get('channels', 64)  # 假设默认值为 64
        self.pos_embed = nn.Linear(2, self.channels)
        self.ConvType = config.get('ConvType','GraphAttention')
        k = config.get('k', 5)

        self.blocks = nn.ModuleList()
        block_configs = config.get('blocks', [
            {'in_channels': self.channels, 'out_channels': self.channels, 'k': k, 'euclidean': True,'ConvType':self.ConvType},
            {'in_channels': self.channels, 'out_channels': self.channels, 'k': k, 'euclidean': False,'ConvType':self.ConvType},
            {'in_channels': self.channels, 'out_channels': self.channels, 'k': k, 'euclidean': False,'ConvType':self.ConvType}
        ])
        for block_config in block_configs:
            self.blocks.append(Block(**block_config))

        self.node_proj = nn.Linear(in_channel, self.channels)
        classifier_config = config.get('classifier', {
            'hidden_channels': self.channels // 2
        })
        self.classifier = Seq(
            nn.Linear(self.channels, classifier_config['hidden_channels']),
            nn.BatchNorm1d(classifier_config['hidden_channels']),
            nn.ReLU(),
            nn.Linear(classifier_config['hidden_channels'], self.nb_classes)
        )

        type_pooling_config = config.get('type_pooling_layer', {
            'n_head': 8
        })
        self.type_pooling_layer = GlobalAttentionBlock(self.channels, **type_pooling_config)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, pos, pseudo_label):
        graph_size = x.shape[2]

        pseudo_label = rearrange(pseudo_label, '(b h) -> b h', h=graph_size * graph_size)
        attn_mask = from_pesudol_label_get_mask(pseudo_label)

        pos_ = self.pos_embed(pos)
        pos_ = rearrange(pos_, '(b h w) d -> b d h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])
        pos = rearrange(pos, '(b h) d -> b h d', b=x.shape[0], h=x.shape[2] * x.shape[2])

        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.node_proj(x)
        z = x
        z = self.type_pooling_layer(z, attn_mask)
        z = rearrange(z, 'b d w -> (b d) w')

        x = rearrange(x, 'b (h w) d -> b d h w', h=graph_size)
        x = x + pos_

        for block in self.blocks:
            x = block(x, coords=pos)

        x = rearrange(x, 'b d h w -> (b h w) d')
        if self.add_pesudo_layer:
            x = 0.8 * x + 0.2 * z
        pred_8 = self.classifier(x)
        return pred_8

if __name__ == '__main__':
    file_path = 'config.json'
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    model = Graph_HNet(config)
    batch_size = 2
    channels = config.get('channels', 64)
    graph_size = 8
    x = torch.randn(batch_size, channels, graph_size, graph_size)
    pos = torch.randn(batch_size * graph_size * graph_size, 2)
    pseudo_label = torch.randn(batch_size * graph_size * graph_size)

    # 前向传播
    pred_8 = model(x, pos, pseudo_label)
    print(pred_8.shape)