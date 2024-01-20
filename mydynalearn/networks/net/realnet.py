import random
import numpy as np
import torch
from scipy.special import comb
from ..util.util import nodeToEdge_matrix,nodeToTriangle_matrix
from mydynalearn.networks.network import Network
from mydynalearn.networks.util.real_network import generate_real_network
import os
import pickle
class Realnet():
    def __init__(self, config):
        self.config = config
        self.DEVICE = self.config.DEVICE
        self.net_config = config.network
        self.NAME = self.net_config.NAME
        self.REALNET_DATA_PATH = self.net_config.REALNET_DATA_PATH
        self.REALNET_SOURCEDATA_FILENAME = self.net_config.REALNET_SOURCEDATA_FILENAME
        self.REALNET_NETDATA_FILENAME = self.net_config.REALNET_NETDATA_FILENAME
        self.MAX_DIMENSION = self.net_config.MAX_DIMENSION
        pass
    def print_log(self,num_indentation=0):
        ''' 对齐输出

        :param num_indentation: 程度
        :return:
        '''
        # 缩进
        num_indentation += 1
        indentation = num_indentation*"\t"
        # 输出内容
        log_items_list = (("network name:",self.NAME),
                    ("network dimension:", self.MAX_DIMENSION),
                    ("number of nodes:", self.NUM_NODES),
                    ("average degree:", self.AVG_K),
                    )

        # 对齐字段宽度
        field_width = int(max([len(log_items[0]) for log_items in log_items_list])) + 2
        # 输出
        for log_items in log_items_list:
            print("{}{:<{}}{}".format(indentation,log_items[0],field_width,log_items[1]))

    def create_net(self):
        self.net_info = self.get_net_info()  # 网络信息
        self._set_net_info()
        self.inc_matrix_adj_info = self._get_adj()  # 关联矩阵
        self.set_inc_matrix_adj_info()
        self.to_device(self.DEVICE)
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

    def save_realnet(self,netdata_file,net_info):
        with open(netdata_file,"wb") as file:
            pickle.dump(net_info,file)

    def load_realnet(self,netdata_file):
        with open(netdata_file, "rb") as file:
            net_info = pickle.load(file)
        return net_info
    def get_net_info(self):
        netsourve_file = os.path.join(self.REALNET_DATA_PATH, self.REALNET_SOURCEDATA_FILENAME)
        netdata_file = os.path.join(self.REALNET_DATA_PATH, self.REALNET_NETDATA_FILENAME)
        if os.path.exists(netdata_file):
            net_info = self.load_realnet(netdata_file)
        else:
            nodes, edges, triangles = generate_real_network(netsourve_file)
            NUM_NODES = nodes.shape[0]
            NUM_EDGES = edges.shape[0]
            NUM_TRIANGLES = triangles.shape[0]
            AVG_K = torch.asarray([2 * len(edges), 3 * len(triangles)]) / NUM_NODES
            net_info = {"nodes": nodes,
                        "edges": edges,
                        "triangles": triangles,
                        "NUM_NODES": NUM_NODES,
                        "NUM_EDGES": NUM_EDGES,
                        "NUM_TRIANGLES": NUM_TRIANGLES,
                        "AVG_K": AVG_K,
                        }
            self.save_realnet(netdata_file,net_info)

        return net_info
    def _get_adj(self):
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix=torch.tensor([[0, 1], [1, 0]],dtype=torch.long)
        edges_inverse = torch.mm(self.edges,inverse_matrix) # 对调两行
        # inc_matrix_0：节点和节点的关联矩阵，即邻接矩阵
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([self.edges.T,edges_inverse.T],dim=1),
                                              values=torch.ones(2*self.NUM_EDGES),
                                              size=(self.NUM_NODES,self.NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(self.nodes, self.edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()

        # inc_matrix_2：节点和高阶边的关联矩阵
        inc_matrix_adj2 = nodeToTriangle_matrix(self.nodes, self.triangles)
        inc_matrix_adj2 = inc_matrix_adj2.to_sparse()
        # 随机断边
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1,
            "inc_matrix_adj2":inc_matrix_adj2
        }
        return inc_matrix_adj_info
    def to_device(self,device):
        self.DEVICE = device
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        self.triangles = self.triangles.to(device)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.NUM_TRIANGLES = self.NUM_TRIANGLES
        self.AVG_K = self.AVG_K

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(device)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(device)
        self.inc_matrix_adj2 = self.inc_matrix_adj2.to(device)
    # def _unpack_net_info(self):
    #     return self.nodes, self.edges, self.triangles, self.NUM_NODES, self.NUM_EDGES, self.NUM_TRIANGLES, self.AVG_K,
    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1, self.inc_matrix_adj2
