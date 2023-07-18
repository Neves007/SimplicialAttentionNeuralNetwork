import loguru

from mydynalearn.networks import *
__networks__ = {
    # Todo: 只改了ER
    "er": er,
    "ba": ba,
    "sc_er": sc_er,
}


def get(config):
    name = config.name
    if name in __networks__:
        net = __networks__[name](config)
        return net
    else:
        loguru.logger.error("there is no network named {}",name)