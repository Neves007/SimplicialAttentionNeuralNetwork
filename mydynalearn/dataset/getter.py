import loguru

from mydynalearn.dataset import *
__networks__ = {
    "ER": graphDataSetLoader,
    "SCER": simplicialDataSetLoader,
    "ToySCER": simplicialDataSetLoader,
    "CONFERENCE": simplicialDataSetLoader,
    "HIGHSCHOOL": simplicialDataSetLoader,
    "HOSPITAL": simplicialDataSetLoader,
    "WORKPLACE": simplicialDataSetLoader,
}


def get(config):
    NAME = config.network.NAME
    if NAME in __networks__:
        DataSetLoader = __networks__[NAME]
        return DataSetLoader
    else:
        raise("there is no network named {}",NAME)