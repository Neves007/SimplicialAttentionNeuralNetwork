from mydynalearn.dataset import *

def get(config,network,dynamics):
    return graph_DynamicDataset(config, network, dynamics)
