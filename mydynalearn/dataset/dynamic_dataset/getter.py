from mydynalearn.dataset import *
# todo:修改其他
__DATA_SET__ = {
    "UAU": DynamicDatasetUAU,
    "CompUAU": DynamicDatasetCompUAU,
    "SCUAU": DynamicDatasetSCUAU,
    "SCCompUAU": DynamicDatasetCompSCUAU,
    "ToySCCompUAU": DynamicDatasetCompSCUAU,
}

def get(config):
    NAME = config.dynamics.NAME
    if NAME in __DATA_SET__:
        dataset = __DATA_SET__[NAME](config)
        return dataset
    else:
        print("there is no dataset named of {}",NAME)
