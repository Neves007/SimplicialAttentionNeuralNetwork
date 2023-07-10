import pickle
import os
import torch
import random
from mydynalearn.networks import er
from mydynalearn.dynamics.util.Simple_Dynamic_weight import Simple_Dynamic_weight
from abc import abstractmethod
from easydict import EasyDict as edict

class graph_DynamicDataset():
    def __init__(self, config,network,dynamics) -> None:
        self.config = config
        self.num_samples = config.num_samples
        self.resampling = 2
        self.device = config.device
        self.network = network
        self.dynamics = dynamics
        assert self.network.maxDimension == self.dynamics.maxDimension == 1
        self.setDynamicInfo(network, dynamics)  # 设置


    def run_dynamicProcess(self):
        self.x0, self.x1 = self.ini_simplexFeatures()  # 初始化单纯型状态
        self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.x0)
        for t in range(self.num_samples):
            x0, y_ob, y_true,neighborSimplexActivation_Matrix = self.dynamic.runOneStep(self.x0, self.incMatrix_AdjAct1, self.network)
            self.x0 = y_ob
            self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.x0)
            self.saveDynamicInfo(t, x0, y_ob, y_true,neighborSimplexActivation_Matrix)

    def run(self, network, dynamic):
        self.setDynamicInfo(network, dynamic) # 设置
        for t in range(self.num_samples):
            self.x0, self.x1 = self.ini_simplexFeatures()  # 初始化单纯型状态
            self.incMatrix_AdjAct1 = self.get_incMatrix_AdjAct1(self.x0,self.x1)
            x0, x1, y_ob, y_true, adjActEdges = self.dynamic.runOneStep(self.x0,self.x1, self.incMatrix_AdjAct1, self.network)
            self.x0 = y_ob
            self.saveDynamicInfo(t, x0, x1, y_ob, y_true,adjActEdges)
        simple_Dynamic_weight = Simple_Dynamic_weight(self.device, self.x0_T, self.adjActEdges_T, self.y_true_T,self.network)
        self.weight = simple_Dynamic_weight.weight

    def saveDynamicInfo(self, t, x0, x1, y_ob, y_true, adjActEdges):
        self.x0_T[t] = x0
        self.x1_T[t] = x1
        self.y_ob_T[t] = y_ob
        self.y_true_T[t] = y_true
        self.adjActEdges_T[t] = adjActEdges

    def setDynamicInfo(self,network, dynamic):
        self.network = network
        self.dynamic = dynamic
        assert self.network.maxDimension == self.dynamic.maxDimension == 1
        self.num_nodes = network.num_nodes
        self.num_edges = network.num_edges
        self.maxDimension = network.maxDimension
        self.infSeeds = int(dynamic.initSeedFraction * self.num_nodes)
        self.num_state = dynamic.num_state
        self.nodeState_map = torch.eye(self.num_state).to(self.device,torch.long)

        self.x0_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.x1_T = torch.ones(self.num_samples, self.num_edges, self.num_state).to(self.config.device,dtype = torch.float)
        self.y_ob_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.y_true_T = torch.ones(self.num_samples, self.num_nodes, self.num_state).to(self.config.device,dtype = torch.float)
        self.adjActEdges_T = torch.ones(self.num_samples, self.num_nodes).to(self.config.device, dtype = torch.long)
        self.simplicesFeature_T = list()

    def get_incMatrix_AdjAct1(self, x0, x1):
        '''
        更具节点状态更新单纯形状态
        '''
        # _expand_x0: 节点特征扩展矩阵
        _expand_x0 = x0.unsqueeze(1).repeat_interleave(self.network.num_edges, dim=1)
        # _globle_incMatrix_lastNumI：边特征减去节点特征，计算剩余节点，I态个数
        _incMatrix_globleLastNumI1 = (x1 - _expand_x0)[:,:,-1]
        # 判断1-单纯型（边）是否为激活态
        _threshold_scAct1 = 1 # 1-simplex中其他节点感染人数阈值
        _incMatrix_globleAct1_indices = torch.where(_incMatrix_globleLastNumI1==_threshold_scAct1)
        _incMatrix_globleAct1_values = _incMatrix_globleLastNumI1[_incMatrix_globleAct1_indices]
        # 节点-激活边 关联矩阵
        _incMatrix_globleAct1_indices = torch.stack(_incMatrix_globleAct1_indices,dim=0)
        _incMatrix_globleAct1 = torch.sparse_coo_tensor(indices=_incMatrix_globleAct1_indices,values=_incMatrix_globleAct1_values,size=_incMatrix_globleLastNumI1.shape)
        incMatrix_AdjAct1 = self.network.incMatrix_adj1*_incMatrix_globleAct1
        return incMatrix_AdjAct1




    def ini_simplexFeatures(self):
        '''
        初始化更新节点状态
        '''
        x0 = torch.zeros(self.num_nodes).to(self.device,torch.long)
        x0 = self.nodeState_map[x0]
        if self.network.toy_network == True:
            infNodes_index = [1,2,5,6]
            x0[infNodes_index] = self.nodeState_map[1]
        else:
            num_infNodes = int(self.num_nodes * self.dynamic.initSeedFraction)
            infNodes_index = random.sample(range(self.num_nodes), num_infNodes)
            x0[infNodes_index] = self.nodeState_map[1]
            # edges_feature: 表示边的特征
            x1 = torch.sum(x0[self.network.edges], dim=-2)
        return x0, x1
    def get_dataset_from_index(self,index):
        dataset = edict({
            "network":self.network,
            "x0_T":self.x0_T[index],
            "x1_T":self.x1_T[index],
            "y_ob_T":self.y_ob_T[index],
            "y_true_T":self.y_true_T[index],
            "adjActEdges_T":self.adjActEdges_T[index],
            "weight":self.weight[index]
        })
        return dataset
    # Todo: splitDataset

    def splitDataset(self,num_test):

        num_train = int((self.num_samples-num_test)/2)
        num_val = num_train
        sample_index = torch.randperm(self.num_samples)
        start = 0
        end = num_train
        train_index = sample_index[start:end]
        start = end
        end += num_val
        val_index = sample_index[start:end]
        start = end
        end += num_test
        test_index = sample_index[start:end]
        # 获取数据集
        trainSet = self.get_dataset_from_index(train_index)
        valSet = self.get_dataset_from_index(val_index)
        testSet = self.get_dataset_from_index(test_index)
        return trainSet, valSet, testSet



