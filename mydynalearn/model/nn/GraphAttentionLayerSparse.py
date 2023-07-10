import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.MultiHeadLinear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


class GATLayer_regular(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        self.Linearlayer1 = nn.Linear(input_size, output_size, bias=bias)
        self.Linearlayer2 = nn.Linear(input_size, output_size, bias=bias)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, features, adj):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        features_i = self.relu(self.Linearlayer1(features))
        features_j = self.relu(self.Linearlayer2(features))

        indices = adj.coalesce().indices()
        self_indices = torch.arange(features_i.shape[0]).unsqueeze(0).repeat_interleave(2,dim=0).to(indices.device)
        indices = torch.cat((indices, self_indices),dim=1)

        a_1 = self.a_1(features_i) # a_1：a*hi
        a_2 = self.a_2(features_j) # a_2：a*hj

        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_1 + a_2.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v)

        # 考虑attention权重的特征。
        output = torch.sparse.mm(attention, features_j)
        # 加上自身嵌入
        output += features_i
        return output

class GraphAttentionLayerSparse(nn.Module):
    def __init__(self, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.layer0_1 = torch.nn.ModuleList([GATLayer_regular(input_size, output_size , bias) for _ in range(heads)])
        self.layer0_2 = torch.nn.ModuleList([GATLayer_regular(output_size, output_size , bias) for _ in range(heads)])
        self.layer0_3 = torch.nn.ModuleList([GATLayer_regular(output_size, output_size , bias) for _ in range(heads)])
        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, x0,x1,network):
        # L0：1阶上部拉普拉斯矩阵
        # L1：【2阶上部拉普拉斯矩阵，2阶下部拉普拉斯矩阵】
        # L2：3阶下部拉普拉斯矩阵
        nodes, edges, incMatrix = network.unpack_NetworkInfo()
        incMatrix_adj0,incMatrix_adj1 = incMatrix

        x0_1 = torch.stack([sat(x0, incMatrix_adj0) for sat in self.layer0_1], dim=1)
        x0_1 = torch.mean(x0_1,dim=1)
        # 输入 X0_1:(2592, 30)
        # 网络：SAT
        # 输出 X0_2:(2592, 30)
        # x0_2 = torch.stack([sat(x0_1, incMatrix_adj0) for sat in self.layer0_2], dim=1)
        # x0_2 = self.leakyrelu(torch.mean(x0_2,dim=1))

        return x0_1