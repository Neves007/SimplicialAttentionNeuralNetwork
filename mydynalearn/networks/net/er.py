import random
import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict
from ..util.util import edge_to_node_matrix

class er():
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
            self.maxDimension = 1
            self.config = config
            self.nodes,self.edges,self.num_nodes,self.num_edges,self.avg_k = self._Create_Network(config.num_nodes,config.avg_k)  # 网络信息
            self.incMatrix_adj0, self.incMatrix_adj1 = self._get_adjInfo()  # 关联矩阵
        self.to_device()
        pass

    # 边界矩阵B
    def _create_edges(self,num_nodes,num_edges):
        edges = set()
        while len(edges) < num_edges:
            edge = random.sample(range(num_nodes), 2)
            edge.sort()
            edges.add(tuple(edge))
        return torch.asarray(list(edges))


    def _Create_Network(self,num_nodes,avg_k):
        assert len(avg_k) == self.maxDimension
        k = avg_k[0]
        nodes = torch.arange(num_nodes)
        num_edges = int(k * num_nodes / 2)
        num_edges = num_edges
        edges = self._create_edges(num_nodes,num_edges)
        avg_k = torch.asarray([2*len(edges)/num_nodes])
        return nodes,edges,num_nodes,num_edges,avg_k

    def _get_adjInfo(self):
        # incMatrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        incMatrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.num_edges),
                                              size=(self.num_nodes,self.num_nodes))
        # incMatrix_1：节点和边的关联矩阵
        incMatrix_adj1 = edge_to_node_matrix(self.edges, self.nodes)
        incMatrix_adj1 = incMatrix_adj1.to_sparse()

        # 随机断边
        return incMatrix_adj0, incMatrix_adj1
    def to_device(self):
        self.nodes = self.nodes.to(self.device)
        self.edges = self.edges.to(self.device)
        self.incMatrix_adj0 = self.incMatrix_adj0.to(self.device)
        self.incMatrix_adj1 = self.incMatrix_adj1.to(self.device)
    def unpack_NetworkInfo(self):
        nodes = self.nodes
        edges = self.edges
        incMatrix = (self.incMatrix_adj0,self.incMatrix_adj1)
        return nodes, edges, incMatrix
