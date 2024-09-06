import torch
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

class SCSF:
    def __init__(self, N, m_tri, device='cuda'):
        """
        初始化无标度单纯复型
        :param N: 最终的节点数
        :param m_tri: 每次添加节点时生成的三角形数
        :param device: 使用的设备 (cpu 或 cuda)
        """
        self.N = N
        self.m_tri = m_tri
        self.m_links = 2 * m_tri
        self.N0 = self.m_links + 1
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.G = nx.Graph()  # 使用 NetworkX 初始化图

        # 初始化初始完全图 (Clique)
        self.G.add_nodes_from(range(self.N0))
        for i in range(self.N0):
            for j in range(i + 1, self.N0):
                self.G.add_edge(i, j)

        # 初始化 Dij，表示边(i, j)所处的三角形数量
        self.Dij = torch.zeros((self.N, self.N), device=self.device)
        for i in range(self.N0):
            for j in range(i + 1, self.N0):
                self.Dij[i, j] = 1
                self.Dij[j, i] = 1

        self._update_Dij()

    def _update_Dij(self):
        """
        更新 Dij 矩阵，Dij 表示每条边所处的三角形数量
        """
        edges = list(self.G.edges)
        for i, j in edges:
            common_neighbors = list(nx.common_neighbors(self.G, i, j))
            count = len(common_neighbors)
            self.Dij[i, j] = count
            self.Dij[j, i] = count

    def _preferential_attachment(self):
        """
        优先附着规则，根据边的权重（边所在三角形的数量）选择边
        """
        # 取上三角矩阵中的元素，因为边是无向的
        indices = torch.triu_indices(self.N, self.N, offset=1, device=self.device)
        tri_weights = self.Dij[indices[0], indices[1]]

        # 累积概率分布，基于边在三角形中的数量
        total_weight = torch.sum(tri_weights)
        if total_weight == 0:
            return None  # 无三角形存在时直接返回
        probabilities = tri_weights / total_weight
        cum_probs = torch.cumsum(probabilities, dim=0)

        # 随机选择 m_tri 条边，根据累积概率
        selected_edges = []
        while len(selected_edges) < self.m_tri:
            rand_vals = torch.rand(self.m_tri, device=self.device)
            for rand_val in rand_vals:
                idx = torch.searchsorted(cum_probs, rand_val)
                i, j = indices[0][idx].item(), indices[1][idx].item()
                if len(set(selected_edges)) == len(selected_edges):  # 确保选择的边不相邻
                    selected_edges.append((i, j))
                if len(selected_edges) == self.m_tri:
                    break

        return selected_edges

    def grow(self):
        """
        按照给定的策略生成无标度单纯复型
        """
        for n in range(self.N0, self.N):
            self.G.add_node(n)

            # 选择现有的边 (i, j)，确保不相邻
            selected_edges = self._preferential_attachment()
            if selected_edges is None:
                break  # 无法继续时退出

            # 添加新的边，生成新的三角形
            for i, j in selected_edges:
                self.G.add_edge(n, i)
                self.G.add_edge(n, j)
                self.G.add_edge(i, j)

            # 更新 Dij 矩阵
            self._update_Dij()

    def degree_distribution(self):
        """
        计算并返回一阶度和二阶度的分布
        :return: 一阶度分布，二阶度分布
        """
        # 计算一阶度分布
        degrees = dict(self.G.degree())
        first_order_degrees = list(degrees.values())

        # 计算二阶度分布（邻居节点的度）
        second_order_degrees = []
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            neighbor_degrees = [degrees[neighbor] for neighbor in neighbors]
            second_order_degrees.extend(neighbor_degrees)

        return first_order_degrees, second_order_degrees

    def plot_log_log_seaborn_scatter(self):
        """
        使用 seaborn 和 log-log 刻度绘制一阶度和二阶度的散点图
        """
        # 获取一阶度和二阶度分布
        first_order_degrees, second_order_degrees = self.degree_distribution()

        # 计算度分布频率
        first_order_count = Counter(first_order_degrees)
        second_order_count = Counter(second_order_degrees)

        # 将度分布转换为概率分布
        total_nodes = sum(first_order_count.values())
        first_order_prob = {k: v / total_nodes for k, v in first_order_count.items()}

        total_neighbors = sum(second_order_count.values())
        second_order_prob = {k: v / total_neighbors for k, v in second_order_count.items()}

        # 转换为 DataFrame 以便于 seaborn 绘图
        first_order_df = pd.DataFrame({
            'Degree': list(first_order_prob.keys()),
            'Probability': list(first_order_prob.values())
        })

        second_order_df = pd.DataFrame({
            'Degree': list(second_order_prob.keys()),
            'Probability': list(second_order_prob.values())
        })

        # 分别绘制一阶度和二阶度的log-log散点图
        plt.figure(figsize=(12, 6))

        # 一阶度散点图
        plt.subplot(1, 2, 1)
        sns.lineplot(data=first_order_df, x='Degree', y='Probability', color='blue')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Log-Log First-order Degree Distribution (Seaborn)")
        plt.xlabel("Degree (log scale)")
        plt.ylabel("Probability (log scale)")

        # 二阶度散点图
        plt.subplot(1, 2, 2)
        sns.lineplot(data=second_order_df, x='Degree', y='Probability', color='green')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Log-Log Second-order Degree Distribution (Seaborn)")
        plt.xlabel("Degree of Neighbors (log scale)")
        plt.ylabel("Probability (log scale)")

        plt.tight_layout()
        plt.show()

# 使用示例
N = 10000  # 总节点数
m_tri = 1  # 每次添加节点时生成的三角形数

scsf = SCSF(N, m_tri)
scsf.grow()
scsf.plot_log_log_seaborn_scatter()  # 绘制log-log散点图（使用Seaborn）
