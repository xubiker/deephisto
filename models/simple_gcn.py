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
from models.graph_hnet import knn,knn_euclidean,MRConv2d
class Block(torch.nn.Module):

    def __init__(self,in_channels, out_channels, k, euclidean=False):
        super(Block, self).__init__()
        self.k = k
        self.euclidean = euclidean
        if euclidean == False:
            self.knn_graph = knn(k=self.k)
        else:
            self.knn_graph = knn_euclidean(k=self.k)
        
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

class GCN_model(torch.nn.Module):
    def __init__(self, channels,nb_classes=5):
        super(GCN_model,self).__init__()
        self.channels = channels
        k=5
        self.block1 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=True)
        self.block2 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=True)
        self.block3 = Block(in_channels=self.channels, out_channels=self.channels, k=k, euclidean=True)
    
        

        self.classifier = Seq(
            nn.Linear(self.channels,self.channels//2),
            nn.ReLU(),
            nn.Linear(self.channels//2,nb_classes)
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x,pos):

        tmp = x
        pos =  rearrange(pos,'(b h) d -> b h d', b=x.shape[0], h=x.shape[2]*x.shape[2]) 
        x = self.block1(x,coords=pos) #4 4
        x = self.block2(x,coords=pos) 
        x = self.block3(x,coords=pos)
        # B x F x N x N 
        x = rearrange(x, 'b d h w -> (b h w) d')
        pred = self.classifier(x)
        return pred
