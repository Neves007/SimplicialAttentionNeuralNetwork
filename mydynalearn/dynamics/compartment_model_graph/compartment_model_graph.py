import copy
from abc import abstractmethod

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
import torch
import random
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter
#  进行一步动力学
class CompartmentModelGraph():
    def __init__(self, config,network):
        self.config = config
        self.dynamics_config = config.dynamics
        self.device = self.config.device
        self.NAME = self.dynamics_config.NAME

        self.MAX_DIMENSION = self.dynamics_config.MAX_DIMENSION
        self.NUM_STATES = self.dynamics_config.NUM_STATES
        self.NODE_FEATURE_MAP = torch.eye(self.NUM_STATES).to(self.device, dtype = torch.long)
        self.SimpleDynamicWeight = weight_getter(self.NAME)
        self.network = network
        self.NUM_NODES = network.NUM_NODES

    def set_x1_from_x0(self):
        self.x1 = torch.sum(self.x0[self.network.edges], dim=-2)

    def init_net_features(self):
        '''
        初始化更新节点状态
        '''

        self._init_x0()
        self.set_x1_from_x0()
    def get_inc_matrix_adjacency_activation(self,inc_matrix_col_feature,_threshold_scAct,target_state):
        '''
        更具节点状态更新1-单纯型状态
        '''
        inc_matrix_col_dim = inc_matrix_col_feature.shape[0]
        # _expand_x0: (节点数，扩展维度，节点状态数)节点特征扩展矩阵
        _expand_x0 = self.x0.unsqueeze(1).repeat_interleave(inc_matrix_col_dim, dim=1)
        # _inc_matrix_feature_without_i：减去节点i后，col_feature的特征，计算剩余节点，target_state态个数
        _inc_matrix_feature_without_i = (inc_matrix_col_feature - _expand_x0)[:,:,self.STATES_MAP[target_state]]
        # 判断1-单纯型（边）是否为激活态
        _inc_matrix_activate_indices = torch.where(_inc_matrix_feature_without_i >= int(_threshold_scAct))
        _inc_matrix_activate_values = _inc_matrix_feature_without_i[_inc_matrix_activate_indices]
        # _inc_matrix_globleAct1：节点-激活边 关联矩阵，全局激活
        _inc_matrix_activate_indices = torch.stack(_inc_matrix_activate_indices,dim=0)
        _inc_matrix_activate = torch.sparse_coo_tensor(indices=_inc_matrix_activate_indices,values=_inc_matrix_activate_values,size=_inc_matrix_feature_without_i.shape)
        # inc_matrix_adj_act_edge：节点-激活边 关联矩阵，即使激活边又相邻
        inc_matrix_activate_adj = self.network.inc_matrix_adj1 * _inc_matrix_activate
        return inc_matrix_activate_adj
    def set_spread_result(self,spread_result):
        self.spread_result = spread_result
    def get_spread_result(self):
        return self.spread_result
    def set_features(self,new_x0, new_x1, **kwargs):
        self.x0 = new_x0
        self.x1 = new_x1
    def get_x1_from_x0(self,x0):
        x1 = torch.sum(x0[self.network.edges], dim=-2)
        return x1

    @abstractmethod
    def _init_x0(self):
        pass