import copy
from abc import abstractmethod

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
import torch
import random
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter
#  进行一步动力学
class CompartmentModelSimplicial():
    def __init__(self, config,network):
        self.config = config
        self.dynamics_config = config.dynamics
        self.device = self.config.device
        self.NAME = self.dynamics_config.NAME

        self.MAX_DIMENSION = self.dynamics_config.MAX_DIMENSION
        self.NUM_STATES = self.dynamics_config.NUM_STATES
        self.NODE_FEATURE_MAP = torch.eye(self.NUM_STATES).to(self.device, dtype = torch.long)
        self.SimpleDynamicWeight = weight_getter(self.NAME)

    def set_network(self,network):
        self.network = network
        self.NUM_NODES = network.NUM_NODES
    def set_x1_from_x0(self):
        self.x1 = torch.sum(self.x0[self.network.edges], dim=-2)
    def get_weight(self,**weight_args):
        simple_dynamic_weight = self.SimpleDynamicWeight(**weight_args)
        weight = simple_dynamic_weight.get_weight()
        return weight
    def set_x2_from_x0(self):
        self.x2 = torch.sum(self.x0[self.network.triangles], dim=-2)

    def get_x1_from_x0(self,x0):
        x1 = torch.sum(x0[self.network.edges], dim=-2)
        return x1
    def get_x2_from_x0(self,x0):
        x2 = torch.sum(x0[self.network.triangles], dim=-2)
        return x2
    def init_net_features(self,network):
        '''
        初始化更新节点状态
        '''
        self.set_network(network)
        self._init_x0()
        self.set_x1_from_x0()
        self.set_x2_from_x0()

    def get_inc_matrix_adjacency_activation(self,inc_matrix_col_feature,_threshold_scAct,target_state,inc_matrix_adj):
        num_target_state_in_simplex = inc_matrix_col_feature[:,self.STATES_MAP[target_state]]
        act_simplex = num_target_state_in_simplex >= _threshold_scAct
        inc_matrix_activate_adj = torch.sparse.FloatTensor.mul(inc_matrix_adj, act_simplex)
        return inc_matrix_activate_adj
    def set_spread_result(self, spread_result):
        self.spread_result = spread_result
    def get_spread_result(self):
        return self.spread_result

    def set_features(self,new_x0, new_x1,new_x2, **kwargs):
        self.x0 = new_x0
        self.x1 = new_x1
        self.x2 = new_x2

    @abstractmethod
    def _init_x0(self):
        pass