import math
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import numpy as np
import torch
from torch import nn
from models.gcn_lib.torch_nn import BasicConv, batched_index_select, act_layer
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

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

    def __init__(self,k):
        super(knn_euclidean, self).__init__()
        self.k = k
        x = torch.arange(8)
        y = torch.arange(8)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coord = torch.stack([grid_x,grid_y],dim=0).to(torch.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.coord = rearrange(coord, 'd h w -> d (h w)').to(device)

    def forward(self,coords):
        x= coords
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=self.k)  #b, n, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1)
        edge_index = torch.stack((nn_idx, center_idx), dim=0)
        return edge_index


class Downsample(nn.Module):

    def __init__(self, in_dim=384, out_dim=384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, Upsample=True,bilinear=True):
        super().__init__()
        self.Upsample = Upsample
        if self.Upsample == True:
            if bilinear == 'bilinear':
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        if self.Upsample == True:
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Block(torch.nn.Module):

    def __init__(self,in_channels, out_channels, k, euclidean=False):
        super(Block, self).__init__()
        self.k = k
        self.euclidean = euclidean
        if euclidean == False:
            self.knn_graph = knn(k=self.k)
        else:
            self.knn_graph = knn_euclidean(k=self.k)
        self.conv = GraphAttention(in_channels, out_channels, k=self.k, act='relu', norm='batch', bias=True)
        #self.conv = GraphSAGE(in_channels, out_channels, act='relu', norm='batch', bias=True)
        #self.conv = EdgeConv2d(in_channels, out_channels, act='relu', norm='batch', bias=True)
        #self.conv = MRConv2d(in_channels, out_channels, act='relu', norm='batch', bias=True)
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


class Graph_HNet(torch.nn.Module):
    def __init__(self, channels,nb_classes=5):

        super(Graph_HNet, self).__init__()
        self.channels = channels
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, 8, 8))
        self.pos_embed = nn.Linear(2,self.channels)
        k = 5
        self.block1 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=True)
        self.block2 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=False)
        self.block3 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=False)
        # self.down2 = Down_block(in_channels=self.channels, out_channels=self.channels, k=5)
        # self.down3 = Down_block(in_channels=self.channels, out_channels=self.channels, k=3)
        # self.classifier = Seq(
        #     nn.Linear(self.channels,self.channels//2),
        #     nn.ReLU(),
        #     nn.Linear(self.channels//2,nb_classes)
        # )
        self.model_init()
        

        self.pred_8 = Seq(nn.Conv2d(self.channels, self.channels//4, 1, bias=True),
                          nn.BatchNorm2d(self.channels//4),
                          act_layer('gelu'),
                          nn.Conv2d(self.channels//4, nb_classes, 1, bias=True))

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x,pos):
        pos_ = self.pos_embed(pos)
        pos_ = rearrange(pos_,'(b h w) d -> b d h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])
        pos =  rearrange(pos,'(b h) d -> b h d', b=x.shape[0], h=x.shape[2]*x.shape[2]) 
        x = x + pos_
        tmp = x

        x = self.block1(x,coords=pos) #4 4
        x = self.block2(x,coords=pos) 
        x = self.block3(x,coords=pos)
        pred_8 = self.pred_8(x)
        pred_8 = rearrange(pred_8, 'b d h w -> (b h w) d')
        # x = rearrange(x, 'b d h w -> (b h w) d')
        # pred_8 = self.classifier(x)
        return pred_8

if __name__ == '__main__':
    
    model = Graph_HNet(2048)
    input = torch.zeros((2,2048,8,8))
    pred_8 = model(input,torch.zeros((128,2)))
    print('here')