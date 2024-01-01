'''重构网络的类
    新建netmanager类，来管理不同平均度的网络
'''
import torch

from mydynalearn.networks import *
from mydynalearn.config import *
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
    "ER": NetworkConfig().er(),
    "SCER": NetworkConfig().sc_er(),
    "CONFERENCE": NetworkConfig().real_scnet_conference(),
    "HIGHSCHOOL": NetworkConfig().real_scnet_high_school(),
    "HOSPITAL": NetworkConfig().real_scnet_hospital(),
    "WORKPLACE": NetworkConfig().real_scnet_workplace(),
}
DATASET_CONFIGER = DatasetConfig().dataset()




def create_network(NETWORK_NAME,AVG_K,AVG_K_DELTA):
    '''
    :param NETWORK_NAME: str
    :param AVG_K: int
    :param AVG_K_DELTA: int
    :return:
    '''
    # get network
    network_config = NETWORK_CONFIG[NETWORK_NAME]
    network = NETWORKS[NETWORK_NAME](network_config)
    network.create_net(AVG_K,AVG_K_DELTA)
    return network




# ER, SCER, ToySCER, CONFERENCE, HIGHSCHOOL, HOSPITAL, WORKPLACE
# 写在networkmanager里面

network_manager = NetworkManager(config=DATASET_CONFIGER)
for AVG_K in DATASET_CONFIGER.AVG_K_LIST:
    AVG_K_DELTA_MIN = 1
    AVG_K_DELTA_MAX = AVG_K/2  # 根据公式 最大二阶度是AVG_K/2
    NUM_K_DELTA = DATASET_CONFIGER.NUM_K  # 网络的个数
    # 每一个一阶平均度
    AVG_K_DELTA_LIST = iter(torch.linspace(AVG_K_DELTA_MIN, AVG_K_DELTA_MAX, NUM_K_DELTA))
    for AVG_K_DELTA in AVG_K_DELTA_LIST:
        # 通过一阶和二阶平均度生成网络
        network = create_network("ER",AVG_K,AVG_K_DELTA)
        # yield network
