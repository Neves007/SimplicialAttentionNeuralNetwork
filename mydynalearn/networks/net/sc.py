import random
import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict

class sc():
    def __init__(self, config,toy_network=False):
        # toy_network = True
        self.toy_network = toy_network
        self.name = config.name
        self.device = config.device
        if toy_network:
            self.num_nodes = 8
            self.maxDimension = 2
            self.simplices_Dict = {
                "1-simplex": torch.tensor([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                              (1, 2), (1, 7),
                              (2, 7), (4, 5), (5, 6)]).to(self.device),
                "2-simplex": torch.tensor([(0, 4, 5), (0, 5, 6), (1, 2, 7)]).to(self.device)
            }
        else:
            self.config = config
            self.num_nodes = config.num_nodes
            self.maxDimension = config.maxDimension
            self.avg_k = np.array(config.avg_k)
            self.simplices_Dict = self._Create_Network()  # 邻接表里存的是单纯型索引
        pass
        self.edge_index, self.simplices_incidence, self.real_K = self._get_net_info()  # 邻接表里存的是单纯型索引

    def get_createSimplex_num(self):
        '''
            通过平均度计算单纯形创建概率
        '''
        # C_matrix单纯形贡献矩阵
        C_matrix = np.zeros((self.maxDimension,self.maxDimension))
        for i in range(self.maxDimension):
            for j in range(i,self.maxDimension):
                C_matrix[i,j] = comb(j+1,i+1)
        C_matrix_inv = np.linalg.pinv(C_matrix)  # 逆
        k_prime = np.dot(C_matrix_inv, self.avg_k)  # 需要生成的单纯形平均度
        # 转换为需要生成的单纯形总个数，乘以N除以单纯形中的节点数(重复的单纯形不算)。
        Num_create_simplices = k_prime * self.num_nodes / np.array([i + 2 for i in range(self.maxDimension)])
        return Num_create_simplices

    def get_simpleices(self):
        simplices_Dict = {"{}-simplex".format(i + 1): set() for i in range(self.maxDimension)}
        for index_dimension in range(self.maxDimension):
            dimension = index_dimension + 1
            num_nodes_in_simplex = dimension + 1
            # 生成simplex
            while len(simplices_Dict["{}-simplex".format(dimension)]) < self.Num_create_simplices[index_dimension]:
                # ER网络
                simplex = random.sample(range(self.num_nodes), num_nodes_in_simplex)
                simplex.sort()
                simplices_Dict["{}-simplex".format(dimension)].add(tuple(simplex))
                for index_lower_dimension in range(0, index_dimension):
                    lower_dimension = index_lower_dimension + 1
                    sub_simplicies = combinations(simplex, lower_dimension + 1)

                    # 添加子单纯形
                    for sub_simplex in sub_simplicies:
                        sub_simplex = list(sub_simplex)
                        sub_simplex.sort()
                        simplices_Dict["{}-simplex".format(lower_dimension)].add(tuple(sub_simplex))
        for index_dimension in range(self.maxDimension):
            dimension = index_dimension + 1
            key = "{}-simplex".format(dimension)
            simplices_Dict[key] = torch.tensor(list(simplices_Dict[key])).to(self.device)
        return simplices_Dict

    def _Create_Network(self):
        self.Num_create_simplices = self.get_createSimplex_num()
        simplices_Dict = self.get_simpleices()

        # 随机断边
        return simplices_Dict

    def _get_simplices_incidence(self,simplices_Dict):
        '''
        转化为关联矩阵
        target_simplices：源节点
        source_simplices：相邻的单纯形
        '''
        simplices_incidence = {}
        for index_dimension in range(self.maxDimension):
            dimension = index_dimension + 1
            key = "{}-simplex".format(dimension)
            target_nodes = []
            source_simplices = []
            for index,simplex in enumerate(simplices_Dict[key]):
                for node in simplex:
                    target_nodes.append(node)
                    source_simplices.append(index)
            data = {"target_nodes":torch.tensor(target_nodes,device=self.device),"source_simplices":torch.tensor(source_simplices,device=self.device)}
            simplices_incidence.update({key:data})
        return edict(simplices_incidence)

    def _get_edge_index(self):
        simplices_1D = self.simplices_Dict['1-simplex']
        edge_index = torch.cat((simplices_1D, simplices_1D[:, [1, 0]]), dim=0).to(self.device)
        return edge_index.T


    def _get_net_info(self):
        edge_index = self._get_edge_index()
        simplices_incidence = self._get_simplices_incidence(self.simplices_Dict)
        # 真实度
        real_K = []
        for index_dimension in range(self.maxDimension):
            dimension = index_dimension + 1
            key = "{}-simplex".format(dimension)
            data = len(simplices_incidence[key].target_nodes)/self.num_nodes
            real_K.append(data)
        return edge_index, simplices_incidence, torch.tensor(real_K)