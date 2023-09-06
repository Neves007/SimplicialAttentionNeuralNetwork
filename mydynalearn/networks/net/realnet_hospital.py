import random
import numpy as np
import torch
from scipy.special import comb
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
from mydynalearn.networks.util.real_network import generate_real_network
import os
class RealnetHospital():
    def __init__(self, net_config):
        self.net_config = net_config
        self.device = net_config.device
        self.REALNET_DATA_PATH = net_config.REALNET_DATA_PATH
        self.REALNET_SOURCEDATA_FILENAME = net_config.REALNET_SOURCEDATA_FILENAME
        self.MAX_DIMENSION = self.net_config.MAX_DIMENSION

        self.net_info = self._create_network()  # 网络信息
        self._set_net_info()
        self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
        self.set_inc_matrix_adj_info()
        self._to_device()
        pass
    def set_inc_matrix_adj_info(self):
        self.inc_matrix_adj0 = self.inc_matrix_adj_info["inc_matrix_adj0"]
        self.inc_matrix_adj1 = self.inc_matrix_adj_info["inc_matrix_adj1"]
        self.inc_matrix_adj2 = self.inc_matrix_adj_info["inc_matrix_adj2"]

    def _set_net_info(self):
        self.nodes = self.net_info["nodes"]
        self.edges = self.net_info["edges"]
        self.triangles = self.net_info["triangles"]
        self.NUM_NODES = self.net_info["NUM_NODES"]
        self.NUM_EDGES = self.net_info["NUM_EDGES"]
        self.NUM_TRIANGLES = self.net_info["NUM_TRIANGLES"]
        self.AVG_K = self.net_info["AVG_K"]

    def _create_network(self):
        file = os.path.join(self.REALNET_DATA_PATH, self.REALNET_SOURCEDATA_FILENAME)
        G,node_neighbors_dict, simplex_list, N = generate_real_network(file)
        nodes = torch.tensor(np.sort(G.nodes),dtype=torch.long)
        edges = torch.tensor(np.asarray(G.edges),dtype=torch.long)
        triangles = torch.tensor(np.asarray(simplex_list),dtype=torch.long)
        NUM_NODES = nodes.shape[0]
        NUM_EDGES = edges.shape[0]
        NUM_TRIANGLES = triangles.shape[0]
        AVG_K = torch.asarray([2*NUM_EDGES,3*NUM_TRIANGLES])/NUM_NODES
        net_info = {"nodes": nodes,
                    "edges": edges,
                    "triangles": triangles,
                    "NUM_NODES": NUM_NODES,
                    "NUM_EDGES": NUM_EDGES,
                    "NUM_TRIANGLES": NUM_TRIANGLES,
                    "AVG_K": AVG_K}
        return net_info
    def _get_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.tensor([[0, 1], [1, 0]],dtype=torch.long)
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
    def _to_device(self):
        self.nodes = self.nodes.to(self.device)
        self.edges = self.edges.to(self.device)
        self.triangles = self.triangles.to(self.device)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.NUM_TRIANGLES = self.NUM_TRIANGLES
        self.AVG_K = self.AVG_K

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(self.device)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(self.device)
        self.inc_matrix_adj2 = self.inc_matrix_adj2.to(self.device)
    def _unpack_net_info(self):
        return self.nodes, self.edges, self.triangles, self.NUM_NODES, self.NUM_EDGES, self.NUM_TRIANGLES, self.AVG_K,
    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1, self.inc_matrix_adj2
