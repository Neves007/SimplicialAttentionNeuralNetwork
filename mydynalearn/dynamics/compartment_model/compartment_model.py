import copy
from abc import abstractmethod

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
import torch
import random
from mydynalearn.dynamics.simple_dynamic_weight.getter import get as weight_getter
#  进行一步动力学
class CompartmentModel():
    def __init__(self, config):
        self.config = config
        self.dynamics_config = config.dynamics
        self.DEVICE = self.config.DEVICE
        self.NAME = self.dynamics_config.NAME

        self.MAX_DIMENSION = self.dynamics_config.MAX_DIMENSION
        self.STATES_MAP = self.dynamics_config.STATES_MAP
        self.NUM_STATES = len(self.STATES_MAP)
        self.NODE_FEATURE_MAP = torch.eye(self.NUM_STATES).to(self.DEVICE, dtype = torch.long)
    def get_transition_state(self,true_tp):
        '''根据迁移概率计算迁移状态

        :param true_tp: 迁移概率
        :return:
        '''
        try:
            states = torch.multinomial(true_tp, num_samples=1, replacement=False).squeeze(1)
            new_x0 = self.NODE_FEATURE_MAP[states]
            return new_x0
        except Exception as e:
            le0_index = torch.where(true_tp.sum(dim=1) <= 0)[0]
            isnan_index = torch.where(torch.isnan(true_tp))[0]
            if len(le0_index)>0:
                print("sum of true_tp <=0\n", le0_index)
                print("state of the node\n", self.x0[le0_index])
            if len(isnan_index)>0:
                print("true_tp is nan\n", isnan_index)
                print("state of the node\n", self.x0[isnan_index])
            raise e


    def calculate_intersection_tensor(self,tensor_a,tensor_b):
        '''计算同时存在tensor_a和tensor_b的元素

        :param tensor_a:
        :param tensor_b:
        :return:
        '''
        # Find the unique elements in each tensor to remove any duplicates
        unique_a = torch.unique(tensor_a)
        unique_b = torch.unique(tensor_b)

        # Find the intersection of the two tensors
        intersection = unique_a[torch.isin(unique_a, unique_b)]
        return intersection
    def print_log(self,num_indentation=0):
        ''' 对齐输出

        :param num_indentation: 程度
        :return:
        '''
        # 缩进
        num_indentation += 1
        indentation = num_indentation*"\t"
        # 输出内容
        log_items_list = (("dynamics name:",self.NAME),)

        # 对齐字段宽度
        field_width = int(max([len(log_items[0]) for log_items in log_items_list])) + 2
        # 输出
        for log_items in log_items_list:
            print("{}{:<{}}{}".format(indentation,log_items[0],field_width,log_items[1]))



    def set_network(self,network):
        self.network = network
        self.NUM_NODES = network.NUM_NODES

    def set_x0(self,x0):
        self.x1 = x0

    def set_x1(self,x1):
        self.x1 = x1

    def set_x2(self,x2):
        self.x2 = x2
    def set_features(self,new_x0, **kwargs):
        self.x0 = new_x0
        x1 = self.get_x1_from_x0(self.x0, self.network)
        self.set_x1(x1)

        if self.network.MAX_DIMENSION==2:
            x2 = self.get_x2_from_x0(self.x0,self.network)
            self.set_x2(x2)

    def get_x1_from_x0(self, x0, network)->'x1':
        x1 = torch.sum(x0[network.edges], dim=-2)
        return x1

    def get_x2_from_x0(self,x0,network):
        x2 = torch.sum(x0[network.triangles], dim=-2)
        return x2

    def init_stateof_network(self):
        '''
        初始化网络状态
        '''
        if hasattr(self, 'network')==False:
            raise AttributeError("network attribute does not exist.")
        self._init_x0()
        x1 = self.get_x1_from_x0(self.x0, self.network)
        self.set_x1(x1)

        if self.network.MAX_DIMENSION==2:
            x2 = self.get_x2_from_x0(self.x0,self.network)
            self.set_x2(x2)

    def get_inc_matrix_adjacency_activation(self,
                                            inc_matrix_col_feature,
                                            _threshold_scAct,
                                            target_state,
                                            inc_matrix_adj):
        '''获取激活邻居关联矩阵


        :param inc_matrix_col_feature: 【x0,x1,x2】关联矩阵列的特征，也就是j的含义。
        :param _threshold_scAct: 该特征是激活态的阈值
        :param target_state: 目标状态
        :param inc_matrix_adj: 网络的关联矩阵
        :return: 激活关联矩阵，行列的含义与inc_matrix_adj相同，行i是节点列j是【节点、边、三角】，元素为1表示i和j相邻且j为激活态
        '''
        num_target_state_in_simplex = inc_matrix_col_feature[:,self.STATES_MAP[target_state]]
        act_simplex = num_target_state_in_simplex >= _threshold_scAct
        # todo：使用稠密矩阵乘法，提高计算效率
        # inc_matrix_activate_adj = torch.mul(inc_matrix_adj, act_simplex)
        inc_matrix_activate_adj = torch.sparse.FloatTensor.mul(inc_matrix_adj, act_simplex)
        return inc_matrix_activate_adj


    def get_spread_result(self):
        return self.spread_result

    def get_weight(self,**weight_args):
        simple_dynamic_weight = self.SimpleDynamicWeight(**weight_args)
        weight = simple_dynamic_weight.get_weight()
        return weight


    @abstractmethod
    def _init_x0(self):
        pass