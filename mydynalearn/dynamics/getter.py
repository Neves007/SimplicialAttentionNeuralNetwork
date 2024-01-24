from mydynalearn.dynamics.compartment_model import *

__DYNAMICS__ = {
    "UAU": UAU,
    "CompUAU": CompUAU,
    "CompUAUCoev": CoevUAU,
    "CoopUAU": CoevUAU,
    "AsymmUAU": CoevUAU,
    "SCUAU": SCUAU,
    "SCCompUAU": SCCoevUAU,
    "SCCoopUAU": SCCoevUAU,
    "SCAsymmUAU": SCCoevUAU,
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME](config)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )
