from mydynalearn.dataset import *

__DataSet__ = {
    "graph_DynamicDataset": graph_DynamicDataset,
    "simplicial_DynamicDataset": simplicial_DynamicDataset
}


def get(config,network,dynamics):
    name = config.name
    if name in __DataSet__:
        dataset = __DataSet__[name](config, network, dynamics)
        return dataset
    else:
        print("there is no dataset named of {}",name)
