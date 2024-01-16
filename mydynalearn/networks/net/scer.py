import random
import numpy as np
import torch
from scipy.special import comb
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
class SCER(Network):
    def __init__(self, net_config):
        super().__init__(net_config)
        pass

    def _set_net_info(self):
        self.nodes = self.net_info["nodes"]
        self.edges = self.net_info["edges"]
        self.triangles = self.net_info["triangles"]
        self.NUM_NODES = self.net_info["NUM_NODES"]
        self.NUM_EDGES = self.net_info["NUM_EDGES"]
        self.NUM_TRIANGLES = self.net_info["NUM_TRIANGLES"]
        self.AVG_K = self.net_info["AVG_K"]

    def set_inc_matrix_adj_info(self):
        self.inc_matrix_adj0 = self.inc_matrix_adj_info["inc_matrix_adj0"]
        self.inc_matrix_adj1 = self.inc_matrix_adj_info["inc_matrix_adj1"]
        self.inc_matrix_adj2 = self.inc_matrix_adj_info["inc_matrix_adj2"]
    def get_createSimplex_num(self,NUM_NODES,AVG_K,AVG_K_DELTA):
        '''
            通过平均度计算单纯形创建概率
        '''
        # C_matrix单纯形贡献矩阵
        C_matrix = np.zeros((self.MAX_DIMENSION,self.MAX_DIMENSION))
        for i in range(self.MAX_DIMENSION):
            for j in range(i,self.MAX_DIMENSION):
                C_matrix[i,j] = comb(j+1,i+1)
        C_matrix_inv = np.linalg.pinv(C_matrix)  # 逆
        k_matrix = np.array([AVG_K, AVG_K_DELTA])
        k_prime = np.dot(C_matrix_inv, k_matrix)  # 需要生成的单纯形平均度
        # 转换为需要生成的单纯形总个数，乘以N除以单纯形中的节点数(重复的单纯形不算)。
        Num_create_simplices = k_prime * NUM_NODES / np.array([i + 2 for i in range(self.MAX_DIMENSION)])
        return Num_create_simplices.astype(np.int16)
    def _create_edges(self, NUM_NODES, NUM_EDGES):
        edges = set()
        while len(edges) < NUM_EDGES:
            edge = random.sample(range(NUM_NODES), 2)
            edge.sort()
            edges.add(tuple(edge))
        return edges

    def _create_triangles(self, NUM_NODES, NUM_TRIANGLES):
        triangles = set()
        while len(triangles) < NUM_TRIANGLES:
            triangle = random.sample(range(NUM_NODES), 3)
            triangle.sort()
            triangles.add(tuple(triangle))
        edges_in_triangles = set()
        for triangle in triangles:
            i, j, k = triangle
            edge1 = [i, j]
            edge1.sort()
            edge2 = [i, k]
            edge2.sort()
            edge3 = [j, k]
            edge3.sort()
            edges_in_triangles.add(tuple(edge1))
            edges_in_triangles.add(tuple(edge2))
            edges_in_triangles.add(tuple(edge3))
        return triangles, edges_in_triangles

    def _merge_edges(self,edges,edges_in_triangles):
        for edge in edges_in_triangles:
            edges.add(edge)
        return edges
    def get_net_info(self):
        # 所需参数
        AVG_K = self.net_config.AVG_K
        AVG_K_DELTA = self.net_config.AVG_K_DELTA
        NUM_NODES = self.net_config.NUM_NODES
        # 根据平均度计算边和三角形的数量
        nodes = torch.arange(NUM_NODES)
        NUM_EDGES, NUM_TRIANGLES = self.get_createSimplex_num(NUM_NODES,AVG_K,AVG_K_DELTA)
        # 生成网络
        edges = self._create_edges(NUM_NODES,NUM_EDGES) # 生成边
        triangles,edges_in_triangles = self._create_triangles(NUM_NODES,NUM_TRIANGLES) # 生成三角形
        edges = self._merge_edges(edges,edges_in_triangles) # 将三角中包含的边的边加入边
        edges = torch.asarray(list(edges)) # 最终的边
        triangles = torch.asarray(list(triangles)) # 最终的三角
        AVG_K = torch.asarray([2*len(edges),3*len(triangles)])/NUM_NODES
        NUM_EDGES = edges.shape[0]
        NUM_TRIANGLES = triangles.shape[0]

        net_info = {"nodes": nodes,
                    "edges": edges,
                    "triangles": triangles,
                    "NUM_NODES": NUM_NODES,
                    "NUM_EDGES": NUM_EDGES,
                    "NUM_TRIANGLES": NUM_TRIANGLES,
                    "AVG_K": AVG_K,
                    }
        return net_info
    def _get_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.NUM_EDGES),
                                              size=(self.NUM_NODES,self.NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()

        inc_matrix_adj2 = nodeToTriangle_matrix(self.nodes, self.triangles)
        inc_matrix_adj2 = inc_matrix_adj2.to_sparse()
        # 随机断边
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1,
            "inc_matrix_adj2":inc_matrix_adj2
        }
        return inc_matrix_adj_info

    def _to_DEVICE(self):
        self.nodes = self.nodes.to(self.DEVICE)
        self.edges = self.edges.to(self.DEVICE)
        self.triangles = self.triangles.to(self.DEVICE)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.NUM_TRIANGLES = self.NUM_TRIANGLES
        self.AVG_K = self.AVG_K

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(self.DEVICE)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(self.DEVICE)
        self.inc_matrix_adj2 = self.inc_matrix_adj2.to(self.DEVICE)

    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1, self.inc_matrix_adj2
