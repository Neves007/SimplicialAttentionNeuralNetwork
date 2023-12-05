
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''

import copy
import torch
from mydynalearn.dynamics.simple_dynamic_weight import *

class SimpleDynamicWeightSCUAU(SimpleDynamicWeight):
    def __init__(self,adj_act_edges,adj_act_triangles,**kwargs):
        super().__init__(**kwargs)
        self.adj_act_edges = adj_act_edges
        self.adj_act_triangles = adj_act_triangles
        self.weight = self.get_weight()

    def get_weight(self):
        # 考虑节点的状态分布
        node_old_state_weight = self.node_state_to_weight(self.old_x0)
        node_edge_topology_element_occurrence, node_triangle_topology_element_occurrence = self.node_topology_element_occurrence()
        node_neighbor_act_edge_element_occurrence, node_neighbor_act_triangle_element_occurrence = self.get_node_neighbor_element_occurrence()

        weight = self.occurrence_to_weight(node_edge_topology_element_occurrence,
                                           node_triangle_topology_element_occurrence,
                                           node_neighbor_act_edge_element_occurrence,
                                           node_neighbor_act_triangle_element_occurrence)
        weight = weight * node_old_state_weight.view(weight.shape)
        weight = self.map_to_range(weight)
        return weight.squeeze()


    def occurrence_to_weight(self,
                             node_edge_topology_element_occurrence,
                             node_triangle_topology_element_occurrence,
                             node_neighbor_act_edge_element_occurrence,
                             node_neighbor_act_triangle_element_occurrence):
        node_edge_topology_weight = torch.pow(self.map_to_range(node_edge_topology_element_occurrence),
                                              -1)
        node_triangle_topology_weight = torch.pow(
            self.map_to_range(node_triangle_topology_element_occurrence), -1)
        node_neighbor_act_edge_weight = torch.pow(
            self.map_to_range(node_neighbor_act_edge_element_occurrence), -1)
        node_neighbor_act_triangle_weight = torch.pow(
            self.map_to_range(node_neighbor_act_triangle_element_occurrence), -1)

        weight = node_edge_topology_weight * \
                 node_triangle_topology_weight * \
                 node_neighbor_act_edge_weight * \
                 node_neighbor_act_triangle_weight
        return weight

    def node_topology_element_occurrence(self):
        node_degree_edge = torch.sum(self.network.inc_matrix_adj0.to_dense(), dim=1)
        node_edge_topology_element_occurrence = self.compute_element_occurrence(node_degree_edge)

        node_degree_triangle = torch.sum(self.network.inc_matrix_adj2.to_dense(), dim=1)
        node_triangle_topology_element_occurrence = self.compute_element_occurrence(node_degree_triangle)
        return node_edge_topology_element_occurrence, node_triangle_topology_element_occurrence

    def get_node_neighbor_element_occurrence(self):
        node_neighbor_act_edge = copy.deepcopy(self.adj_act_edges.to(torch.float))
        node_neighbor_act_edge_element_occurrence = self.compute_element_occurrence(node_neighbor_act_edge)

        node_neighbor_act_triangle = copy.deepcopy(self.adj_act_triangles.to(torch.float))
        node_neighbor_act_triangle_element_occurrence = self.compute_element_occurrence(node_neighbor_act_triangle)

        return node_neighbor_act_edge_element_occurrence, node_neighbor_act_triangle_element_occurrence
