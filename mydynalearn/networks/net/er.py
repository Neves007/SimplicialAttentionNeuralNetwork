import random
import torch
from ..util.util import nodeToEdge_matrix
from mydynalearn.networks.network import Network
class ER(Network):
    def __init__(self, net_config):
        super().__init__(net_config)
        pass

    def _set_net_info(self):
        self.nodes = self.net_info["nodes"]
        self.edges = self.net_info["edges"]
        self.NUM_NODES = self.net_info["NUM_NODES"]
        self.NUM_EDGES = self.net_info["NUM_EDGES"]
        self.AVG_K = self.net_info["AVG_K"]

    def set_inc_matrix_adj_info(self):
        self.inc_matrix_adj0 = self.inc_matrix_adj_info["inc_matrix_adj0"]
        self.inc_matrix_adj1 = self.inc_matrix_adj_info["inc_matrix_adj1"]

    # 边界矩阵B
    def _create_edges(self,NUM_NODES,NUM_EDGES):
        edges = set()
        while len(edges) < NUM_EDGES:
            edge = random.sample(range(NUM_NODES), 2)
            edge.sort()
            edges.add(tuple(edge))
        return torch.asarray(list(edges))

    def _create_network(self):
        NUM_NODES = self.net_config.NUM_NODES
        AVG_K = self.net_config.AVG_K
        assert len(AVG_K) == self.MAX_DIMENSION
        k = AVG_K[0]
        nodes = torch.arange(NUM_NODES)
        NUM_EDGES = int(k * NUM_NODES / 2)
        NUM_EDGES = NUM_EDGES
        edges = self._create_edges(NUM_NODES, NUM_EDGES)
        AVG_K = torch.asarray([2 * len(edges) / NUM_NODES])

        net_info = {"nodes": nodes,
                    "edges": edges,
                    "NUM_NODES": NUM_NODES,
                    "NUM_EDGES": NUM_EDGES,
                    "AVG_K": AVG_K}
        return net_info

    def _get_adj(self):
        nodes, edges, NUM_NODES, NUM_EDGES, AVG_K = self._unpack_net_info()
        # inc_matrix_0：节点和节点的关联矩阵
        # 先对边进行预处理，无相边会有问题。
        inverse_matrix = torch.asarray([[0, 1], [1, 0]])
        edges_inverse = torch.mm(edges, inverse_matrix)  # 对调两行
        inc_matrix_adj0 = torch.sparse_coo_tensor(indices=torch.cat([edges.T, edges_inverse.T], dim=1),
                                                 values=torch.ones(2 * NUM_EDGES),
                                                 size=(NUM_NODES, NUM_NODES))
        # inc_matrix_1：节点和边的关联矩阵
        inc_matrix_adj1 = nodeToEdge_matrix(nodes, edges)
        inc_matrix_adj1 = inc_matrix_adj1.to_sparse()
        inc_matrix_adj_info = {
            "inc_matrix_adj0":inc_matrix_adj0,
            "inc_matrix_adj1":inc_matrix_adj1
        }
        # 随机断边
        return inc_matrix_adj_info

    def _to_device(self):
        self.nodes = self.nodes.to(self.device)
        self.edges = self.edges.to(self.device)
        self.NUM_NODES = self.NUM_NODES
        self.NUM_EDGES = self.NUM_EDGES
        self.AVG_K = self.AVG_K

        self.inc_matrix_adj0 = self.inc_matrix_adj0.to(self.device)
        self.inc_matrix_adj1 = self.inc_matrix_adj1.to(self.device)
    def _unpack_net_info(self):
        return self.nodes, self.edges, self.NUM_NODES, self.NUM_EDGES, self.AVG_K
    def _unpack_inc_matrix_adj_info(self):
        return self.inc_matrix_adj0, self.inc_matrix_adj1