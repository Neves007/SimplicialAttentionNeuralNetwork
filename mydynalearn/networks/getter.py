import loguru

from mydynalearn.networks import *
__networks__ = {
    "ER": ER,
    "SCER": SCER,
    "ToySCER": ToySCER,
}


def get(config):
    NAME = config.network.NAME
    if NAME in __networks__:
        net = __networks__[NAME](config.network)
        return net
    else:
        loguru.logger.error("there is no network named {}",NAME)