import pickle
import os
import torch
import random
from mydynalearn.networks import er
from mydynalearn.dynamics.util.Simple_Dynamic_weight import Simple_Dynamic_weight
from abc import abstractmethod

class DynamicDataset_Graph():
    def __init__(self, config) -> None:
        self.config = config
        self.num_samples = config.num_samples
        self.resampling = 2
        self.device = config.device


    def run_dynamicProcess(self, network, dynamic):
        self.setDynamicInfo(network, dynamic) # 设置
        self.ini_nodeFeatures()  # 初始化节点状态
        self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.nodesFeature)
        for t in range(self.num_samples):
            nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix = self.dynamic.runOneStep(self.nodesFeature, self.source_simplicesActivation_Dict, self.network)
            self.nodesFeature = y_ob
            self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.nodesFeature)
            self.saveDynamicInfo(t, nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix)

    def run(self, network, dynamic):
        self.setDynamicInfo(network, dynamic) # 设置
        for t in range(self.num_samples):
            self.ini_nodeFeatures()  # 初始化节点状态
            self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.nodesFeature)
            nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix = self.dynamic.runOneStep(self.nodesFeature, self.incMatrix_AdjAct1, self.network)
            self.nodesFeature = y_ob
            self.saveDynamicInfo(t, nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix)
        simple_Dynamic_weight = Simple_Dynamic_weight(self.device, self.nodeFeature_T, self.neighborSimplexActivation_T,self.y_true_T)
        self.weight = simple_Dynamic_weight.weight

    def saveDynamicInfo(self, t, nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix):
        self.nodeFeature_T[t] = nodeFeature
        self.y_ob_T[t] = y_ob
        self.y_true_T[t] = y_true
        self.neighborSimplexActivation_T[t] = neighborSimplexActivation_Matrix

    def setDynamicInfo(self,network, dynamic):
        self.network = network
        self.dynamic = dynamic
        assert self.network.maxDimension == self.dynamic.maxDimension == 1
        self.num_nodes = network.num_nodes
        self.maxDimension = network.maxDimension
        self.infSeeds = int(dynamic.initSeedFraction * self.num_nodes)
        self.num_state = dynamic.num_state
        self.nodeState_map = torch.eye(self.num_state).to(self.device,torch.long)

        self.nodeFeature_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.y_ob_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.y_true_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.neighborSimplexActivation_T = torch.ones(self.num_samples, self.maxDimension, self.num_nodes, 2).to(self.config.device,dtype = torch.long)
        self.simplicesFeature_T = list()

    def get_incMatrix_AdjAct1(self, nodesFeature):
        '''
        更具节点状态更新单纯形状态
        '''
        # edges_feature: 表示边的特征
        edges_feature = torch.sum(nodesFeature[self.network.edges], dim=-2)
        # _expand_nodesFeature: 节点特征扩展矩阵
        _expand_nodesFeature = nodesFeature.unsqueeze(1).repeat_interleave(self.network.num_edges, dim=1)
        # _globle_incMatrix_lastNumI：边特征减去节点特征，计算剩余节点，I态个数
        _incMatrix_globleLastNumI1 = (edges_feature - _expand_nodesFeature)[:,:,-1]
        # 判断1-单纯型（边）是否为激活态
        _threshold_scAct1 = 1 # 1-simplex中其他节点感染人数阈值
        _incMatrix_globleAct1_indices = torch.where(_incMatrix_globleLastNumI1==_threshold_scAct1)
        _incMatrix_globleAct1_values = _incMatrix_globleLastNumI1[_incMatrix_globleAct1_indices]
        # 节点-激活边 关联矩阵
        _incMatrix_globleAct1_indices = torch.stack(_incMatrix_globleAct1_indices,dim=0)
        _incMatrix_globleAct1 = torch.sparse_coo_tensor(indices=_incMatrix_globleAct1_indices,values=_incMatrix_globleAct1_values,size=_incMatrix_globleLastNumI1.shape)
        incMatrix_AdjAct1 = self.network.incMatrix_adj1*_incMatrix_globleAct1
        return incMatrix_AdjAct1




    def ini_nodeFeatures(self):
        '''
        初始化更新节点状态
        '''
        self.nodesFeature = torch.zeros(self.num_nodes).to(self.device,torch.long)
        self.nodesFeature = self.nodeState_map[self.nodesFeature]
        if self.network.toy_network == True:
            infNodes_index = [1,2,5,6]
            self.nodesFeature[infNodes_index] = self.nodeState_map[1]
        else:
            num_infNodes = int(self.num_nodes * self.dynamic.initSeedFraction)
            infNodes_index = random.sample(range(self.num_nodes), num_infNodes)
            self.nodesFeature[infNodes_index] = self.nodeState_map[1]
