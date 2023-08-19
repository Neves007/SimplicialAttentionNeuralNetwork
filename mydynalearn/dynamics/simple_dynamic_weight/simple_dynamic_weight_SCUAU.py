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

class SimpleDynamicWeightSCUAU:
    def __init__(self,device,old_x0,new_x0,adj_act_edges,adj_act_triangles,network,dynamics):
        self.device = device
        self.old_x0 = old_x0
        self.new_x0 = new_x0
        self.adj_act_edges = adj_act_edges
        self.adj_act_triangles = adj_act_triangles
        self.network = network
        self.dynamics = dynamics

        self.T = old_x0.shape[0]
        self.weight = self.get_weight()

    def get_weight(self):
        # 考虑节点的状态分布
        node_old_state_vec = self.get_node_state_vec(self.old_x0)
        node_new_state_vec = self.get_node_state_vec(self.new_x0)
        node_topology_vec_edge, node_topology_vec_triangle = self.get_node_topology_vec()
        node_neighbor_vec_edges, node_neighbor_vec_triangles = self.get_node_neighbor_vec()
        node_vec = torch.cat((node_old_state_vec,
                              node_new_state_vec,
                              node_topology_vec_edge,
                              node_topology_vec_triangle,
                              node_neighbor_vec_edges,
                              node_neighbor_vec_triangles),dim=-1)
        weight = self.vec_to_weight(node_vec)
        return weight.to(self.device)


    def get_node_state_vec(self,x):
        '''
            原文是统计所有时间步状态的情况来算分布。
            这里只用初始种子来做分布。原理是一样的。
            返回一个(T,NUM_NODES)维向量，元素表示节点状态为xi的概率。
        '''

        return x.cpu()

    def get_node_topology_vec(self):
        # todo:higher order topology
        node_degree_edge = torch.sum(self.network.inc_matrix_adj0.to_dense(), dim=1)
        node_degree_edge /= torch.max(node_degree_edge)

        node_degree_triangle = torch.sum(self.network.inc_matrix_adj2.to_dense(), dim=1)
        node_degree_triangle /= torch.max(node_degree_triangle)

        node_topology_vec_edge = torch.unsqueeze(node_degree_edge,dim=-1)
        node_topology_vec_triangle = torch.unsqueeze(node_degree_triangle,dim=-1)
        return node_topology_vec_edge.cpu(), node_topology_vec_triangle.cpu()

    def get_node_neighbor_vec(self):
        node_neighbor_vec_edges = self.adj_act_edges.to(torch.float)
        node_neighbor_vec_edges /= torch.max(node_neighbor_vec_edges)
        node_neighbor_vec_edges = torch.unsqueeze(node_neighbor_vec_edges,dim=-1)

        node_neighbor_vec_triangles = self.adj_act_triangles.to(torch.float)
        node_neighbor_vec_triangles /= torch.max(node_neighbor_vec_triangles)
        node_neighbor_vec_triangles = torch.unsqueeze(node_neighbor_vec_triangles,dim=-1)
        return node_neighbor_vec_edges.cpu(), node_neighbor_vec_triangles.cpu()
    def vec_to_weight(self,node_vec):
        origin_shape = node_vec.shape
        node_vec_plan = node_vec.view(-1,origin_shape[-1])
        distances = torch.cdist(node_vec_plan, node_vec_plan, p=2)
        distances.fill_diagonal_(0)
        # 找到每个节点最近的5个节点的索引
        k = 10
        nearest_values, nearest_indices =torch.topk(distances, k=k+1,largest=False)  # largest=False 表示找最小的 k 个值
        weight = nearest_values.sum(dim=-1)
        return weight