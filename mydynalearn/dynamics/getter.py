from .compartmentModel import *

__dynamics__ = {
    "sis": sis,
    "sir": sir,
    "sis_sc": sis,
}


def get(config):
    name = config.name
    if name in __dynamics__:
        return __dynamics__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
