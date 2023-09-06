import torch
from scipy.stats import binom
import numpy as np
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''
import copy
import torch
from mydynalearn.dynamics.simple_dynamic_weight import *

class SimpleDynamicWeightCompUAU(SimpleDynamicWeight):
    def __init__(self,adj_A1_act_edges,adj_A2_act_edges,**kwargs):
        super().__init__(**kwargs)
        self.adj_A1_act_edge = adj_A1_act_edges
        self.adj_A2_act_edge = adj_A2_act_edges
        self.weight = self.get_weight()

    def get_weight(self):
        # 考虑节点的状态分布
        node_old_state_weight = self.node_state_to_weight(self.old_x0)
        node_edge_topology_element_occurrence = self.node_topology_element_occurrence()
        node_neighbor_A1_edge_element_occurrence,node_neighbor_A2_edge_element_occurrence = self.get_node_neighbor_element_occurrence()

        weight = self.occurrence_to_weight(node_edge_topology_element_occurrence,
                                           node_neighbor_A1_edge_element_occurrence,
                                           node_neighbor_A2_edge_element_occurrence,
                                           )
        weight = weight * node_old_state_weight.view(weight.shape)
        weight = self.map_to_range(weight)
        return weight.squeeze()

    def occurrence_to_weight(self,
                             node_edge_topology_element_occurrence,
                             node_neighbor_A1_edge_element_occurrence,
                             node_neighbor_A2_edge_element_occurrence,):
        node_edge_topology_weight = torch.pow(self.map_to_range(node_edge_topology_element_occurrence),
                                              -1)
        node_neighbor_A1_edge_weight = torch.pow(
            self.map_to_range(node_neighbor_A1_edge_element_occurrence), -1)
        node_neighbor_A2_edge_weight = torch.pow(
            self.map_to_range(node_neighbor_A2_edge_element_occurrence), -1)

        weight = node_edge_topology_weight * \
                 node_neighbor_A1_edge_weight * \
                 node_neighbor_A2_edge_weight
        return weight

    def node_topology_element_occurrence(self):
        node_degree_edge = torch.sum(self.network.inc_matrix_adj0.to_dense(), dim=1)
        node_edge_topology_element_occurrence = self.compute_element_occurrence(node_degree_edge)
        return node_edge_topology_element_occurrence

    def get_node_neighbor_element_occurrence(self):
        node_neighbor_A1_edge = copy.deepcopy(self.adj_A1_act_edge.to(torch.float))
        node_neighbor_A1_edge_element_occurrence = self.compute_element_occurrence(node_neighbor_A1_edge)

        node_neighbor_A2_edge = copy.deepcopy(self.adj_A2_act_edge.to(torch.float))
        node_neighbor_A2_edge_element_occurrence = self.compute_element_occurrence(node_neighbor_A2_edge)
        return node_neighbor_A1_edge_element_occurrence,node_neighbor_A2_edge_element_occurrence