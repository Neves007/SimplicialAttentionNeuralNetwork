import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# todo: 补充参数
    # nodes 节点序列
    # edges 边序列
    # triangles 三角序列
    # NUM_NODES 节点数
    # NUM_EDGES 边数
    # NUM_TRIANGLES 三角数
    # AVG_K 一阶平均度
    # AVG_K_DELTA 二阶平均度
# todo：相邻关系
    # inc_matrix_adj_info
# todo: 管理过程
    # 保存和读取
    # 输出

class SCSF():
    def __init__(self,net_config):
        self.net_config = net_config
        self.set_attr(net_config)
        pass

    def set_attr(self, attributes):
        '''
        批量设置属性
        :param attributes:
        :return:
        '''
        for key, value in attributes.items():
            setattr(self, key, value)

    def compute_pcum_from_plink(self, plink):
        pcum = torch.cat([torch.tensor([0], device=self.DEVICE), torch.cumsum(plink[:, 2] / torch.sum(plink[:, 2]), dim=0)])
        return pcum
    
    def _init_network(self):
        # Create initial adjacency matrix on cuda
        A0 = torch.ones((self.N0, self.N0), device=self.DEVICE)
        A0 = A0 - torch.diag(torch.diag(A0))
        A = torch.zeros((self.NUM_NODES, self.NUM_NODES), device=self.DEVICE)
        A[:self.N0, :self.N0] = A0

        # 三角参与矩阵，元素表示边（i，j）上都多少个三角形
        Dij = A * (A @ A)

        # plink：Dij的稀疏表示
        # Extract upper triangular part of Dij and find non-zero elements
        TDij = torch.triu(Dij)
        non_zero_indices = torch.nonzero(TDij > 0)
        plink = torch.cat([non_zero_indices, TDij[non_zero_indices[:, 0], non_zero_indices[:, 1]].view(-1, 1)], dim=1)

        # 由plink的度分布决定的累积概率
        pcum = self.compute_pcum_from_plink(plink)

        attr = {
            "A": A,
            "Dij":Dij,
            "plink":plink,
            "pcum":pcum,
        }
        self.set_attr(attr)
    
    def _add_a_node(self, new_node_i):
        self.A[new_node_i, new_node_i] = 0
        l = len(self.plink)
        isw = 0

        # 附着
        # 将所有节点附着到ntri个连边上
        while isw == 0:
            # Generate ntri random numbers
            dummy = torch.rand(self.ntri, device=self.DEVICE)

            # 根据累积概率向量 (pcum) 和随机生成的值 (dummy)，确定哪些边被选中。
            diffs = self.pcum.view(-1, 1) - dummy.view(1, -1)  # 计算累积概率 pcum 和随机生成的值 dummy 之间的差异。
            temp = torch.diff(torch.sign(diffs), dim=0)  # 沿着维度 0（行）计算相邻元素之间的差异。
            idx = torch.nonzero(temp != 0)[:, 0]  # 确定符号变化的行索引（即找到随机值在哪个累积概率区间中）。

            # 确保选择的ntri个连边里没有重复节点
            if len(torch.unique(self.plink[idx, :2])) == 2 * self.ntri:
                isw = 1
        return idx
    
    def _update_adj_info(self, new_node_i, idx):
        
        for link_index in range(len(idx)):
            inode = int(self.plink[idx[link_index], 0].item())
            jnode = int(self.plink[idx[link_index], 1].item())

            self.A[new_node_i, inode] = 1
            self.A[new_node_i, jnode] = 1
            self.A[inode, new_node_i] = 1
            self.A[jnode, new_node_i] = 1

            # Update Dij
            self.Dij[new_node_i, inode] = 1
            self.Dij[new_node_i, jnode] = 1
            self.Dij[inode, new_node_i] = 1
            self.Dij[jnode, new_node_i] = 1
            self.Dij[inode, jnode] += 1
            self.Dij[jnode, inode] += 1

            # 更新 plink：添加两条新的边，并且连边k上多一个三角形
            self.plink = torch.cat([self.plink, torch.tensor([[new_node_i, inode, 1], [new_node_i, jnode, 1]], device=self.DEVICE)], dim=0)
            self.plink[idx[link_index], 2] = self.Dij[inode, jnode].item()

                

    def _add_new_nodes(self):
        # 添加新节点
        for new_node_i in range(self.N0, self.NUM_NODES):
            idx = self._add_a_node(new_node_i)
            self._update_adj_info(new_node_i, idx)
            pcum = self.compute_pcum_from_plink(self.plink)
            self.__setattr__("pcum", pcum)
        
    def build(self):
        self._init_network()
        self._add_new_nodes()


    def get_net_info(self):
        # 所需参数
        self.N0 = 3
        super().__setattr__("self.N0", self.N0)
        # 根据平均度计算边和三角形的数量
        nodes = torch.arange(self.NUM_NODES)
        self.build()



# Example of usage
net_config = {
    "NAME": 'SCSF',
    "NUM_NODES": 1000,
    "MAX_DIMENSION": 2,
    "ntri": 1,
    "DEVICE": "cpu",
}
net = SCSF(net_config)

net_info = net.get_net_info()


def plot_degree_distributions(A, Dij):
    # Calculate the degree distribution
    degrees = A.sum(dim=0).cpu().numpy()
    triangle_degrees = Dij.sum(dim=0).cpu().numpy()

    # Calculate the frequency of each degree
    degree_counts = np.bincount(degrees.astype(int))
    triangle_degree_counts = np.bincount(triangle_degrees.astype(int))

    # Filter out zero counts for log-log plot
    nonzero_degrees = np.nonzero(degree_counts)[0]
    nonzero_degree_counts = degree_counts[nonzero_degrees]

    nonzero_triangle_degrees = np.nonzero(triangle_degree_counts)[0]
    nonzero_triangle_degree_counts = triangle_degree_counts[nonzero_triangle_degrees]

    # Plot the degree distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.loglog(nonzero_degrees, nonzero_degree_counts, 'bo-', markersize=5)
    plt.title('Degree Distribution (Log-Log)')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')

    # Plot the triangle degree distribution
    plt.subplot(1, 2, 2)
    plt.loglog(nonzero_triangle_degrees, nonzero_triangle_degree_counts, 'ro-', markersize=5)
    plt.title('Triangle Degree Distribution (Log-Log)')
    plt.xlabel('Triangle Degree')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
plot_degree_distributions(net.A, net.Dij)