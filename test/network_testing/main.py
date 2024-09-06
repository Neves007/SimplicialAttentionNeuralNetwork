'''真实网络结构分析
    现象：真实网络预测的不准确
    推测：原因可能是训练集网络的异质性可能不够
    实验目的：分析真实网络结构，从而为实验参数提供依据
'''
import torch

from mydynalearn.networks import *
from mydynalearn.config import Config
import matplotlib.pyplot as plt


NETWORKS = {
    "ER": ER,
    "SCER": SCER,
    "ToySCER": ToySCER,
    "CONFERENCE": Realnet,
    "HIGHSCHOOL": Realnet,
    "HOSPITAL": Realnet,
    "WORKPLACE": Realnet,
}

NETWORK_CONFIG = {
    "ER": Config.get_config_network().ER,
    "SCER": Config.get_config_network().SCER,
    "CONFERENCE": Config.get_config_network().CONFERENCE,
    "HIGHSCHOOL": Config.get_config_network().HIGHSCHOOL,
    "HOSPITAL": Config.get_config_network().HOSPITAL,
    "WORKPLACE": Config.get_config_network().WORKPLACE,
}

def get_simple_net_info(NETWORK_NAME):
    # get network
    network_config = NETWORK_CONFIG[NETWORK_NAME]
    network = NETWORKS[NETWORK_NAME](network_config)
    network.create_net()

    # get number of node/edge
    num_nodes = len(network.nodes)
    num_edges = len(network.edges)

    # get network degree list
    K_list = network.inc_matrix_adj0.to_dense().sum(dim=1)  # 一阶度列表

    # get max/min degere
    max_K = K_list.max()
    min_K = K_list.min()
    K_list_distribution = torch.bincount(K_list.to(torch.int))
    print(
        "REAL NETWORK NAME: {} \nnum_nodes: {} \nnum edges: {}\nMAX DEGREE: {} \nmin degree: {}\n\n".format(
            NETWORK_NAME, num_nodes, num_edges, max_K, min_K, K_list_distribution))
    draw_degree_distribution(K_list_distribution, NETWORK_NAME, K_NAME="Degree")

def get_higher_net_info(NETWORK_NAME):
    # get network
    network_config = NETWORK_CONFIG[NETWORK_NAME]
    network = NETWORKS[NETWORK_NAME](network_config)
    network.create_net()

    # get number of node/edge
    num_nodes = len(network.nodes)
    num_edges = len(network.edges)

    # get network degree list
    K_list = network.inc_matrix_adj0.to_dense().sum(dim=1)  # 一阶度列表
    K_delta_list = network.inc_matrix_adj2.to_dense().sum(dim=1)  # 二阶度列表

    # get max/min degere
    max_K = K_list.max()
    min_K = K_list.min()
    max_K_delta = K_delta_list.max()
    min_K_delta = K_delta_list.min()

    # get degree distribution
    K_list_distribution = torch.bincount(K_list.to(torch.int))
    K_delta_list_distribution = torch.bincount(K_delta_list.to(torch.int))

    print(
        "REAL NETWORK NAME: {} \nnum_nodes: {} \n num edges: {}\nMAX DEGREE: {} \nmax triangle degree: {} \nmin degree: {}  \nmin triangle degree: {}\n\n".format(
            NETWORK_NAME, num_nodes, num_edges, max_K, max_K_delta, min_K, min_K_delta, K_list_distribution))
    draw_degree_distribution(K_list_distribution, NETWORK_NAME, K_NAME="Degree")
    draw_degree_distribution(K_delta_list_distribution, NETWORK_NAME, K_NAME="Triangle degree")







def draw_degree_distribution(K_list_distribution,NETWORK_NAME,K_NAME):
    K_list_distribution = K_list_distribution.cpu()
    # 设置x轴标签
    x_labels = torch.arange(len(K_list_distribution))
    # 绘制柱状图
    plt.bar(x_labels, K_list_distribution)
    # 设置标题和坐标标签
    plt.title('{1} distribution of {0} network'.format(NETWORK_NAME,K_NAME))
    plt.xlabel('Degree')
    plt.ylabel('Counts')
    # 显示图形
    plt.show()

# ER, SCER, ToySCER, CONFERENCE, HIGHSCHOOL, HOSPITAL, WORKPLACE
get_simple_net_info("ER")
get_higher_net_info("SCER")
get_higher_net_info("CONFERENCE")
get_higher_net_info("HIGHSCHOOL")
get_higher_net_info("HOSPITAL")
get_higher_net_info("WORKPLACE")


