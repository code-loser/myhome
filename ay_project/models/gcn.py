import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu , allow_zero_in_degree=True)
        )

        self.layers.append(
            dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )

        self.layers.append(
            dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True)
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

            if l != len(self.layers) - 1:
                h = self.dropout(h)

        return h