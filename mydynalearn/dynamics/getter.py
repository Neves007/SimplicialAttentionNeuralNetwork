from mydynalearn.dynamics.compartment_model_graph import *
from mydynalearn.dynamics.compartment_model_simplicial import *

__DYNAMICS__ = {
    "UAU": UAU,
    "CompUAU": CompUAU,
    "SCUAU": SCUAU,
    "SCCompUAU": SCCompUAU,
    "ToySCCompUAU": ToySCCompUAU,
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME](config)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )
