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

class Simple_Dynamic_weight:
    def __init__(self,device,x_T,adjActEdges_T,y_true_T,network):
        self.device = device
        self.cpu_device = torch.device('cpu')
        self.y_true_T = y_true_T.to(self.cpu_device)
        self.x_class_T:'T,num_nodes' = TP_to_class(x_T).to(self.cpu_device)
        self.adjActEdges_T = adjActEdges_T.to(self.cpu_device)

        self.network = network
        self.num_nodes = self.x_class_T.shape[1]
        self.T = self.x_class_T.shape[0]
        self.weight = self.get_weight()

    def shrink_tensor(self,w,min=0.1,max=1):
        w_min = w.min()
        w_max = w.max()
        new_min = min
        new_max = max
        if w_min == w_max:
            size = min*torch.ones(w.shape)
        else:
            size = (w - w_min) / (w_max - w_min) * (new_max - new_min) + new_min
        return size

    def get_weight(self):
        # 考虑节点的状态分布
        nodeState_distribution = self.get_nodeState_distribution()
        # 考虑节点一阶度分布
        # k1_distribution_T = self.get_1simplexDegree_distribution()
        # 考虑节点的邻居单纯形状态分布
        # neighborActivation_1simplex_distribution = self.get_NeighborActivation_1simplex_distribution()
        # 权重
        total_distribution = nodeState_distribution
        weight = torch.pow(total_distribution,exponent=-0.5)
        # lim_max_weight = torch.quantile(weight, 0.9, interpolation='nearest', dim=1).max()
        # weight = torch.clamp(weight, max=lim_max_weight)
        return weight.to(self.device)

    def uniqueCount_2_overallCount(self, overall_value, unique, unique_count):
        # 分布
        unique_dis = unique_count/unique_count.sum()
        # unique转换成overall_value的维度
        unique = unique.repeat(list(overall_value.shape) + [1])
        # overall_value中的值转换为unique中的索引
        overall_value_2_unique_index = torch.where(unique == overall_value[..., None])[2].reshape(overall_value.shape)
        # 通过索引找到对应的分布
        overall_value_2_overallCount = unique_dis[overall_value_2_unique_index]
        return overall_value_2_overallCount

    def get_nodeState_distribution(self):
        '''
            原文是统计所有时间步状态的情况来算分布。
            这里只用初始种子来做分布。原理是一样的。
            返回一个(T,num_nodes)维向量，元素表示节点状态为xi的概率。
        '''
        unique_x,count_x = torch.unique(self.x_class_T,return_counts=True)
        nodeState_distribution_T = self.uniqueCount_2_overallCount(self.x_class_T,unique_x,count_x)
        return nodeState_distribution_T

