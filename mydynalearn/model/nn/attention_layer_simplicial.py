


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.multi_head_linear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


class SATLayer_regular(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.a_1 = nn.Linear(output_size, 1, bias=bias)
        self.a_2 = nn.Linear(output_size, 1, bias=bias)
        self.a_3 = nn.Linear(output_size, 1, bias=bias)
        self.a_4 = nn.Linear(output_size, 1, bias=bias)
        self.linear_layer1 = nn.Linear(input_size, output_size, bias=bias)
        self.linear_layer2 = nn.Linear(input_size, output_size, bias=bias)
        self.linear_layer3 = nn.Linear(input_size, output_size, bias=bias)
        self.linear_layer4 = nn.Linear(input_size, output_size, bias=bias)
        self.agg_weight = nn.Parameter(torch.randn(3))
        self.LinearAgg = nn.Linear(2*output_size, output_size, bias=bias)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def attention_agg(self, xj, a_i,a_j,inc_matrix_adj):
        indices = inc_matrix_adj.coalesce().indices()

        # a_1 + a_2.T：e矩阵
        # v：e矩阵，有效e值
        attention_v = torch.sigmoid(a_i + a_j.T)[indices[0, :], indices[1, :]]
        # e矩阵转为稀疏矩阵
        attention = torch.sparse_coo_tensor(indices, attention_v,size=inc_matrix_adj.shape)

        # 考虑attention权重的特征。
        output = torch.sparse.mm(attention, xj)
        return output

    def forward(self,network,dynamics, x0, x1, x2=None):
        """
        features : n * m dense matrix of feature vectors
        adj : n * n  sparse signed orientation matrix
        output : n * k dense matrix of new feature vectors
        """
        if x2==None:
            incMatrix_adj0, incMatrix_adj1 = network._unpack_inc_matrix_adj_info()
            xi_0 = self.relu(self.linear_layer1(x0))
            xj_0 = self.relu(self.linear_layer2(x0))


            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj

            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)

            x0 = xi_0 + agg0
        else:
            incMatrix_adj0, incMatrix_adj1, incMatrix_adj2 = network._unpack_inc_matrix_adj_info()
            xi_0 = self.relu(self.linear_layer1(x0))
            xj_0 = self.relu(self.linear_layer2(x0))
            xj_1 = self.relu(self.linear_layer3(x1))
            xj_2 = self.relu(self.linear_layer4(x2))

            ai_0 = self.a_1(xi_0)  # a_1：a*hi
            aj_0 = self.a_2(xj_0)  # a_2：a*hj
            aj_1 = self.a_2(xj_1)  # a_2：a*hj
            aj_2 = self.a_2(xj_2)  # a_2：a*hj
            agg0 = self.attention_agg(xj_0, ai_0, aj_0, incMatrix_adj0)
            agg1 = self.attention_agg(xj_1, ai_0, aj_1, incMatrix_adj1)
            agg2 = self.attention_agg(xj_2, ai_0, aj_2, incMatrix_adj2)

            x0 = xi_0 + agg0 + agg1 + agg2
        return x0

class SimplexAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, heads, concat, bias=True):
        super().__init__()
        self.layer0_1 = torch.nn.ModuleList([SATLayer_regular(input_size, output_size, bias) for _ in range(heads)])


    def forward(self, **kwargs):
        x0_1 = torch.stack([sat(**kwargs) for sat in self.layer0_1])
        x0_1 = torch.mean(x0_1, dim=0)


        return x0_1