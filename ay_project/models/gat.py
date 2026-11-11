import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import cProfile
import sys, os
import math

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads=4):
        super().__init__()

        self.layers = nn.ModuleList()

        
        self.layers.append(
            dglnn.GATConv(
                in_size,
                hid_size // num_heads,   
                num_heads,               
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        self.layers.append(
            dglnn.GATConv(
                hid_size,                
                hid_size // num_heads,
                num_heads,
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        
        self.layers.append(
            dglnn.GATConv(
                hid_size,
                out_size,
                num_heads=1,
                feat_drop=0.5,
                attn_drop=0.5,
                allow_zero_in_degree=True
            )
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # h shape = (N, num_heads, out_dim)
            h = h.flatten(1)

            if l != len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)

        return h