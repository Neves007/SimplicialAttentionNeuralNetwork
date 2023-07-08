import random
import networkx as nx
import numpy as np
import torch
from itertools import combinations
from scipy.special import comb

class ba():
    def __init__(self, config):
        self.config = config
        self.num_nodes = config.num_nodes
        self.avg_k = config.avg_k
        self.BE_eadge_p = config.BE_eadge_p
        self.net= self._Create_Network()  # 邻接表里存的是单纯型索引
        self.edge_index, self.real_K= self._get_net_info(self.net)  # 邻接表里存的是单纯型索引
        pass


    def _Create_Network(self):
        net = nx.barabasi_albert_graph(self.num_nodes, self.avg_k)
        # 随机断边
        num_broken_edges = int(len(net.edges())*self.BE_eadge_p)
        BE_edge_bunch = random.sample(net.edges(),num_broken_edges)
        net.remove_edges_from(BE_edge_bunch)
        return net
    def _get_net_info(self,net):
        # 邻接矩阵
        edge_index = torch.tensor(list(net.edges()))
        # 无向图
        edge_index = torch.cat((edge_index,edge_index[:,(-1,0)]),dim=0).transpose(1,0).to(self.config.device)
        # 真实度
        __degree = dict(nx.degree(net))
        real_K = sum(__degree.values())/len(net.nodes)
        return edge_index, real_K