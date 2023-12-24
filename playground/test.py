'''以迭代器的方式生成网络

'''
import torch
import numpy as np
class Network:
    def __init__(self):
        self.k_min = 1
        self.k_max = 4
        self.num_k = 4
        self.k_list = iter(torch.linspace(self.k_min,self.k_max,self.num_k))

    def __iter__(self):
        return self

    def __next__(self):
        k = next(self.k_list)
        print("create netwrok with degree {:f}".format(k))
        return self


network_manager = Network()
iter_networks = iter(network_manager)

next(iter_networks)