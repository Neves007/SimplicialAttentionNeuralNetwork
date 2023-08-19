import torch
from scipy.stats import binom
import numpy as np
import copy
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''

from mydynalearn.dynamics.simple_dynamic_weight import *
import torch
from collections import Counter
class SimpleDynamicWeightSCCompUAU:
    def __init__(self,device,old_x0,new_x0,adj_A1_act_edge,adj_A2_act_edge,adj_A1_act_triangle,adj_A2_act_triangle,network,dynamics):
        self.device = device
        self.old_x0 = old_x0
        self.new_x0 = new_x0
        self.adj_A1_act_edge = adj_A1_act_edge
        self.adj_A2_act_edge = adj_A2_act_edge
        self.adj_A1_act_triangle = adj_A1_act_triangle
        self.adj_A2_act_triangle = adj_A2_act_triangle
        self.network = network
        self.dynamics = dynamics

        self.T = old_x0.shape[0]
        self.weight = self.get_weight()
    def get_weight(self):
        # 考虑节点的状态分布
        node_old_state_weight = self.node_state_to_weight(self.old_x0)
        node_edge_topology_element_occurrence, node_triangle_topology_element_occurrence = self.node_topology_element_occurrence()
        node_neighbor_A1_edge_element_occurrence, node_neighbor_A2_edge_element_occurrence, node_neighbor_A1_triangle_element_occurrence, node_neighbor_A2_triangle_element_occurrence = self.get_node_neighbor_element_occurrence()

        weight = self.occurrence_to_weight(node_edge_topology_element_occurrence,
                                           node_triangle_topology_element_occurrence,
                                           node_neighbor_A1_edge_element_occurrence,
                                           node_neighbor_A2_edge_element_occurrence,
                                           node_neighbor_A1_triangle_element_occurrence,
                                           node_neighbor_A2_triangle_element_occurrence)
        weight = weight * node_old_state_weight.view(weight.shape)
        weight /= weight.sum()
        return weight.squeeze()

    def map_to_range(self, array, min=0, max=2):
        '''
        将数值隐射到到target_min_value，target_max_value之间
        '''
        min_value = torch.min(array)
        max_value = torch.max(array)
        normalized_values = min + (array - min_value) * (max - min) / (max_value - min_value)
        return normalized_values

    def occurrence_to_weight(self,
                             node_edge_topology_element_occurrence,
                             node_triangle_topology_element_occurrence,
                             node_neighbor_A1_edge_element_occurrence,
                             node_neighbor_A2_edge_element_occurrence,
                             node_neighbor_A1_triangle_element_occurrence,
                             node_neighbor_A2_triangle_element_occurrence):
        node_edge_topology_weight = torch.exp(-self.map_to_range(node_edge_topology_element_occurrence,min=0,max=1))
        node_triangle_topology_weight = torch.exp(-self.map_to_range(node_triangle_topology_element_occurrence,min=0,max=1))
        node_neighbor_A1_edge_weight = torch.exp(-self.map_to_range(node_neighbor_A1_edge_element_occurrence,min=0,max=1))
        node_neighbor_A2_edge_weight = torch.exp(-self.map_to_range(node_neighbor_A2_edge_element_occurrence,min=0,max=1))
        node_neighbor_A1_triangle_weight = torch.exp(-self.map_to_range(node_neighbor_A1_triangle_element_occurrence,min=0,max=1))
        node_neighbor_A2_triangle_weight = torch.exp(-self.map_to_range(node_neighbor_A2_triangle_element_occurrence,min=0,max=1))

        weight = node_edge_topology_weight * \
                 node_triangle_topology_weight * \
                 node_neighbor_A1_edge_weight * \
                 node_neighbor_A2_edge_weight * \
                 node_neighbor_A1_triangle_weight * \
                 node_neighbor_A2_triangle_weight
        return weight

    def node_state_to_weight(self,x):
        '''
            原文是统计所有时间步状态的情况来算分布。
            这里只用初始种子来做分布。原理是一样的。
            返回一个(T,NUM_NODES)维向量，元素表示节点状态为xi的概率。
        '''
        num_node_state = x.sum(dim=0)
        num_node_state_prob = (x * num_node_state).sum(dim=1) / self.network.NUM_NODES
        num_node_state_weight = torch.exp(-self.map_to_range(num_node_state_prob, min=0, max=1))
        return num_node_state_weight

    def node_topology_element_occurrence(self):
        # todo:higher order topology
        node_degree_edge = torch.sum(self.network.inc_matrix_adj0.to_dense(), dim=1)
        node_edge_topology_element_occurrence = self.compute_element_occurrence(node_degree_edge)

        node_degree_triangle = torch.sum(self.network.inc_matrix_adj2.to_dense(), dim=1)
        node_triangle_topology_element_occurrence = self.compute_element_occurrence(node_degree_triangle)
        return node_edge_topology_element_occurrence, node_triangle_topology_element_occurrence

    def get_node_neighbor_element_occurrence(self):
        node_neighbor_A1_edge = copy.deepcopy(self.adj_A1_act_edge.to(torch.float))
        node_neighbor_A1_edge_element_occurrence = self.compute_element_occurrence(node_neighbor_A1_edge)

        node_neighbor_A2_edge = copy.deepcopy(self.adj_A2_act_edge.to(torch.float))
        node_neighbor_A2_edge_element_occurrence = self.compute_element_occurrence(node_neighbor_A2_edge)

        node_neighbor_A1_triangle = copy.deepcopy(self.adj_A1_act_triangle.to(torch.float))
        node_neighbor_A1_triangle_element_occurrence = self.compute_element_occurrence(node_neighbor_A1_triangle)

        node_neighbor_A2_triangle = copy.deepcopy(self.adj_A1_act_triangle.to(torch.float))
        node_neighbor_A2_triangle_element_occurrence = self.compute_element_occurrence(node_neighbor_A2_triangle)

        return node_neighbor_A1_edge_element_occurrence,node_neighbor_A2_edge_element_occurrence,node_neighbor_A1_triangle_element_occurrence,node_neighbor_A2_triangle_element_occurrence

    # def compute_element_occurrence(self, array):
    #     # 计算范围内的元素个数
    #     array = array.view(len(array), 1)
    #     neighbor_counts = torch.zeros_like(array)
    #     distances_matrix = abs(array.T - array)
    #     distances_counter = Counter(distances_matrix.view(-1).tolist())
    #     most_common_width = distances_counter.most_common(1)[0][0]
    #     n_min = array.min()
    #     n_max = array.max()
    #
    #     width = 4*most_common_width
    #     for i in range(len(array)):
    #         n = array[i]
    #         if (n - width)>=n_min and (n + width)<=n_max:
    #             count = ((array >= n - width) & (array <= n + width)).sum()
    #         elif n - width<n_min:
    #             count = (array <= n_min+2*width).sum()
    #         elif (n + width)>n_max:
    #             count = (array >= n_max-2*width).sum()
    #         neighbor_counts[i] = count
    #     return neighbor_counts

    def compute_element_occurrence(self,array):
        array = array.view(len(array), 1)
        neighbor_counts = torch.zeros_like(array)
        distances_matrix = torch.abs(array.T - array)
        distances_values, distances_count = torch.unique(distances_matrix,return_counts=True)
        most_common_width = distances_values[distances_count.argmax()]
        n_min = array.min()
        n_max = array.max()

        width = 4 * most_common_width

        mask1 = ((array - width) >= n_min).squeeze()
        mask2 = ((array + width) <= n_max).squeeze()
        mask3 = ((array - width) < n_min).squeeze()
        mask4 = ((array + width) > n_max).squeeze()

        count1 = ((array >= (array.T - width)) & (array <= (array.T + width))).sum(dim=1).to(torch.float)
        count2 = (array <= (n_min + 2 * width)).sum().to(torch.float)
        count3 = (array >= (n_max - 2 * width)).sum().to(torch.float)

        neighbor_counts[mask1 & mask2] = count1[mask1 & mask2].view(-1,1)
        neighbor_counts[mask3] = count2
        neighbor_counts[mask4] = count3

        return neighbor_counts