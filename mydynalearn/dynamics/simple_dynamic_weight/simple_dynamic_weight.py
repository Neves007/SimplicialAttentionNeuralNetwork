import torch
from scipy.stats import binom
import numpy as np
'''
公式（7）和公式（16），计算Loss前面的那个权重的。
避免训练数据不平衡导致训练不稳定。
'''


import torch

class SimpleDynamicWeight():
    def __init__(self,device,old_x0,new_x0,network,dynamics,**kwargs):
        assert len(kwargs) == 0
        self.device = device
        self.dynamics = dynamics
        self.network = network
        self.old_x0 = old_x0
        self.new_x0 = new_x0
        self.T = old_x0.shape[0]

    def compute_element_occurrence(self, array):
        array = array.view(len(array), 1)
        neighbor_counts = torch.zeros_like(array)
        distances_matrix = torch.abs(array.T - array)
        distances_values, distances_count = torch.unique(distances_matrix, return_counts=True)
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

        neighbor_counts[mask1 & mask2] = count1[mask1 & mask2].view(-1, 1)
        neighbor_counts[mask3] = count2
        neighbor_counts[mask4] = count3

        return neighbor_counts
    def map_to_range(self, array, min=1, max=3):
        '''
        将数值隐射到到target_min_value，target_max_value之间
        '''
        min_value = torch.min(array)
        max_value = torch.max(array)
        normalized_values = min + (array - min_value) * (max - min) / (max_value - min_value)
        return normalized_values

    def node_state_to_weight(self,x):
        num_node_state = x.sum(dim=0)
        num_node_state_prob = (x * num_node_state).sum(dim=1) / self.network.NUM_NODES
        num_node_state_weight = torch.pow(self.map_to_range(num_node_state_prob),-1)
        return num_node_state_weight