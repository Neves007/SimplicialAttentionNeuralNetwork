


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.MultiHeadLinear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


class SATLayer_regular(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        self.Linearlayer1 = nn.Linear(input_size, output_size, bias=bias)
        self.Linearlayer2 = nn.Linear(input_size, output_size, bias=bias)
        self.LinearAgg = nn.Linear(4*output_size, output_size, bias=bias)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def attention_agg(self, xi, xj, incMatrix_adj):
        indices = incMatrix_adj.coalesce().indices()

        a_1 = self.a_1(xi)  # a_1：a*hi
        a_2 = self.a_2(xj)  # a_2：a*hj

        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_1 + a_2.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v,size=incMatrix_adj.shape)

        # 考虑attention权重的特征。
        output = torch.sparse.mm(attention, xj)
        return output

    def forward(self, x0, x1, x2, incMatrix_adj0, incMatrix_adj1, incMatrix_adj2):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        xi_0 = self.relu(self.Linearlayer1(x0))
        xj_0 = self.relu(self.Linearlayer1(x0))
        xj_1 = self.relu(self.Linearlayer1(x1))
        xj_2 = self.relu(self.Linearlayer1(x2))

        agg0 = self.attention_agg(xi_0, xj_0, incMatrix_adj0)
        agg1 = self.attention_agg(xi_0, xj_1, incMatrix_adj1)
        agg2 = self.attention_agg(xi_0, xj_2, incMatrix_adj2)

        x0 = self.LinearAgg(torch.cat((xi_0,agg0,agg1,agg2),dim=1))
        return x0

class SimplexAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.layer0_1 = torch.nn.ModuleList([SATLayer_regular(input_size, output_size, bias) for _ in range(heads)])
        self.layer0_2 = torch.nn.ModuleList([SATLayer_regular(output_size, output_size, bias) for _ in range(heads)])
        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, x0_1, x1, x2, network):
        # L0：1阶上部拉普拉斯矩阵
        # L1：【2阶上部拉普拉斯矩阵，2阶下部拉普拉斯矩阵】
        # L2：3阶下部拉普拉斯矩阵
        nodes, edges, triangles, incMatrix = network.unpack_SimplicialInfo()
        incMatrix_adj0, incMatrix_adj1, incMatrix_adj2 = incMatrix

        x0_1 = torch.stack([sat(x0_1, x1, x2, incMatrix_adj0, incMatrix_adj1, incMatrix_adj2) for sat in self.layer0_1])
        x0_1 = torch.mean(x0_1, dim=0)
        x1_1 = torch.sparse.mm(incMatrix_adj1.T, x0_1)
        x2_1 = torch.sparse.mm(incMatrix_adj2.T, x0_1)
        #
        # x0_2 = torch.stack([sat(x0_1, x1_1, x2_1, incMatrix_adj0, incMatrix_adj1, incMatrix_adj2) for sat in self.layer0_2])
        # x0_2 = torch.mean(x0_2, dim=0)
        # x1_2 = torch.sparse.mm(incMatrix_adj1.T, x0_2)
        # x2_2 = torch.sparse.mm(incMatrix_adj2.T, x0_2)

        return x0_1