import random
import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb
from easydict import EasyDict as edict
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix

class sc_er():
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
            self.maxDimension = 2
            self.config = config
            self.nodes, self.edges, self.triangles, self.num_nodes, self.num_edges, self.num_triangles, self.avg_k = self._Create_SC(config.num_nodes, config.avg_k)
            self.incMatrix_adj0, self.incMatrix_adj1, self.incMatrix_adj2 = self._get_adjInfo()  # 关联矩阵
        self.to_device()
        pass
    def get_createSimplex_num(self,num_nodes,avg_k):
        '''
            通过平均度计算单纯形创建概率
        '''
        # C_matrix单纯形贡献矩阵
        C_matrix = np.zeros((self.maxDimension,self.maxDimension))
        for i in range(self.maxDimension):
            for j in range(i,self.maxDimension):
                C_matrix[i,j] = comb(j+1,i+1)
        C_matrix_inv = np.linalg.pinv(C_matrix)  # 逆
        k_prime = np.dot(C_matrix_inv, avg_k)  # 需要生成的单纯形平均度
        # 转换为需要生成的单纯形总个数，乘以N除以单纯形中的节点数(重复的单纯形不算)。
        Num_create_simplices = k_prime * num_nodes / np.array([i + 2 for i in range(self.maxDimension)])
        return Num_create_simplices.astype(np.int)

    # 边界矩阵B
    def _create_edges(self,num_nodes,num_edges):
        edges = set()
        while len(edges) < num_edges:
            edge = random.sample(range(num_nodes), 2)
            edge.sort()
            edges.add(tuple(edge))
        return edges
    def _create_triangles(self,num_nodes,num_triangles):
        triangles = set()
        while len(triangles) < num_triangles:
            triangle = random.sample(range(num_nodes), 3)
            triangle.sort()
            triangles.add(tuple(triangle))
        edges_in_triangles = set()
        for triangle in triangles:
            i,j,k = triangle
            edge1 = [i,j]
            edge1.sort()
            edge2 = [i,k]
            edge2.sort()
            edge3 = [j,k]
            edge3.sort()
            edges_in_triangles.add(tuple(edge1))
            edges_in_triangles.add(tuple(edge2))
            edges_in_triangles.add(tuple(edge3))
        return triangles,edges_in_triangles

    def _merge_edges(self,edges,edges_in_triangles):
        for edge in edges_in_triangles:
            edges.add(edge)
        return edges
    def _Create_SC(self, num_nodes, avg_k):
        assert len(avg_k) == self.maxDimension
        k = avg_k[0]
        nodes = torch.arange(num_nodes)
        num_edges, num_triangles = self.get_createSimplex_num(num_nodes,avg_k)
        edges = self._create_edges(num_nodes,num_edges)
        triangles,edges_in_triangles = self._create_triangles(num_nodes,num_triangles)
        edges = self._merge_edges(edges,edges_in_triangles)
        edges = torch.asarray(list(edges))
        triangles = torch.asarray(list(triangles))
        avg_k = torch.asarray([2*len(edges),3*len(triangles)])/num_nodes
        num_edges = edges.shape[0]
        num_triangles = triangles.shape[0]
        return nodes,edges,triangles,num_nodes,num_edges,num_triangles,avg_k

    def _get_adjInfo(self):
        # incMatrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        incMatrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.num_edges),
                                              size=(self.num_nodes,self.num_nodes))
        # incMatrix_1：节点和边的关联矩阵
        incMatrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        incMatrix_adj1 = incMatrix_adj1.to_sparse()

        incMatrix_adj2 = nodeToTriangle_matrix(self.nodes, self.triangles)
        incMatrix_adj2 = incMatrix_adj2.to_sparse()
        # 随机断边
        return incMatrix_adj0, incMatrix_adj1, incMatrix_adj2
    def to_device(self):
        self.nodes = self.nodes.to(self.device)
        self.edges = self.edges.to(self.device)
        self.triangles = self.triangles.to(self.device)
        self.incMatrix_adj0 = self.incMatrix_adj0.to(self.device)
        self.incMatrix_adj1 = self.incMatrix_adj1.to(self.device)
        self.incMatrix_adj2 = self.incMatrix_adj2.to(self.device)
    def unpack_NetworkInfo(self):
        nodes = self.nodes
        edges = self.edges
        triangles = self.triangles
        incMatrix = (self.incMatrix_adj0,self.incMatrix_adj1)
        return nodes, edges, incMatrix
    def unpack_SimplicialInfo(self):
        nodes = self.nodes
        edges = self.edges
        triangles = self.triangles
        incMatrix = (self.incMatrix_adj0,self.incMatrix_adj1,self.incMatrix_adj2)
        return nodes, edges,triangles, incMatrix
