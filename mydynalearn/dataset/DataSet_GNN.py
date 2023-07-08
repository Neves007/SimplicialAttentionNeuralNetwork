import pickle
import os
import torch
import random
from mydynalearn.networks import er
from mydynalearn.dynamics.util.Simple_Dynamic_weight import Simple_Dynamic_weight
from abc import abstractmethod

class DataSet_GNN():
    def __init__(self, config) -> None:
        self.config = config
        self.num_samples = config.num_samples
        self.resampling = 2
        self.device = config.device


    def run_dynamicProcess(self, network, dynamic):
        self.setDynamicInfo(network, dynamic) # 设置
        self.ini_nodeFeatures()  # 初始化节点状态
        self.source_simplicesActivation_Dict = self.getSourceSimplicesActivation()
        for t in range(self.num_samples):
            nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix = self.dynamic.runOneStep(self.nodesState, self.source_simplicesActivation_Dict, self.network)
            self.nodesState = y_ob
            self.source_simplicesActivation_Dict = self.getSourceSimplicesActivation()
            self.saveDynamicInfo(t, nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix)

    def run(self, network, dynamic):
        self.setDynamicInfo(network, dynamic) # 设置
        for t in range(self.num_samples):
            self.ini_nodeFeatures()  # 初始化节点状态
            self.source_simplicesActivation_Dict = self.getSourceSimplicesActivation()
            nodeFeature, y_ob, y_true,neighborSimplexActivation_Matrix = self.dynamic.runOneStep(self.nodesState, self.source_simplicesActivation_Dict, self.network)
            self.nodesState = y_ob
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
        assert self.network.maxDimension == self.dynamic.maxDimension
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

    def getSourceSimplicesActivation(self):
        '''
        更具节点状态更新单纯形状态
        '''
        simplexActivationMap = torch.eye(2).to(self.device)
        source_simplicesActivation_Dict = {}
        for index_dimension in range(self.network.maxDimension):
            dimension = index_dimension + 1
            key = "{}-simplex".format(dimension)

            cur_simplex_Dict = self.network.simplices_Dict[key]
            cur_simplex_incidence = self.network.simplices_incidence[key]
            source_simplices = cur_simplex_incidence['source_simplices']
            target_nodes = cur_simplex_incidence['target_nodes']
            # 源单纯形包含的节点
            source_simplices_nodes = cur_simplex_Dict[source_simplices]
            source_simplices_nodeState = self.nodesState[source_simplices_nodes].sum(dim=-2)
            target_nodesState = self.nodesState[target_nodes]
            # simplexActivationEvidence单纯形状态判断依据。单纯形中所有节点状态，除去自身节点状态
            source_simplicesActivationEvidence = source_simplices_nodeState-target_nodesState
            simplexActivation_threshold = dimension
            # 出当前节点的其他节点，I态节点数量大于阈值为激活态
            source_simplicesActivation = simplexActivationMap[(source_simplicesActivationEvidence[:,1]>=simplexActivation_threshold).to(torch.long)]
            source_simplicesActivation_Dict.update({key:source_simplicesActivation})
        return source_simplicesActivation_Dict




    def ini_nodeFeatures(self):
        '''
        初始化更新节点状态
        '''
        self.nodesState = torch.zeros(self.num_nodes).to(self.device,torch.long)
        self.nodesState = self.nodeState_map[self.nodesState]
        if self.network.toy_network == True:
            infNodes_index = [1,2,5,6]
            self.nodesState[infNodes_index] = self.nodeState_map[1]
        else:
            num_infNodes = int(self.num_nodes * self.dynamic.initSeedFraction)
            infNodes_index = random.sample(range(self.num_nodes), num_infNodes)
            self.nodesState[infNodes_index] = self.nodeState_map[1]
