import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GCN1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) # 第一层卷积
        self.conv2 = GCNConv(hidden_channels, out_channels) # 第二层卷积

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index)) # 第一层激活
        x = F.dropout(x, p=0.5, training=self.training) # 第一层dropout
        x = self.conv2(x, edge_index) # 第二层输出
        return F.log_softmax(x, dim=1) # 对数softmax