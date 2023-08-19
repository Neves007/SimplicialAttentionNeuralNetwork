import torch
from scipy.stats import binom
import numpy as np
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''

import matplotlib.pyplot as plt
from mydynalearn.util.myplot import myplot
from mydynalearn.transformer import *
from scipy import stats
import torch

class SimpleDynamicWeightUAU:
    def __init__(self,device,old_x0,new_x0, adj_act_edges,network,dynamics):
        self.device = device
        self.dynamics = dynamics
        self.network = network
        self.old_x0 = old_x0
        self.new_x0 = new_x0
        self.T = old_x0.shape[0]
        self.adj_act_edges = adj_act_edges
        self.weight = self.get_weight()

    def get_weight(self):
        # 考虑节点的状态分布
        node_state_vec = self.get_node_state_vec()
        node_topology_vec = self.get_node_topology_vec()
        node_neighbor_vec = self.get_node_neighbor_vec()
        node_vec = torch.cat((node_state_vec,
                            node_topology_vec,
                            node_neighbor_vec),dim=-1)
        weight = self.vec_to_weight(node_vec)
        return weight.to(self.device)


    def get_node_state_vec(self):
        '''
            原文是统计所有时间步状态的情况来算分布。
            这里只用初始种子来做分布。原理是一样的。
            返回一个(T,NUM_NODES)维向量，元素表示节点状态为xi的概率。
        '''

        return self.x_T.cpu()

    def get_node_topology_vec(self):
        node_degree_edge = torch.sum(self.network.inc_matrix_adj0.to_dense(), dim=1)
        node_degree_edge /= torch.max(node_degree_edge)
        node_topology_vec = torch.unsqueeze(node_degree_edge,dim=-1)
        return node_topology_vec.cpu()

    def get_node_neighbor_vec(self):
        node_neighbor_vec = self.adj_act_edges.to(torch.float)
        node_neighbor_vec /= torch.max(node_neighbor_vec)
        node_neighbor_vec = torch.unsqueeze(node_neighbor_vec,dim=-1)
        return node_neighbor_vec.cpu()
    def vec_to_weight(self,node_vec):
        distances = torch.cdist(node_vec, node_vec)

        # 找到每个节点最近的5个节点的索引
        k = 10
        nearest_values, nearest_indices =torch.topk(distances, k=k+1,largest=False)  # largest=False 表示找最小的 k 个值
        return nearest_values.sum(dim=-1)