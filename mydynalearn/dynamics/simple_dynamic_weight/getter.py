from mydynalearn.dynamics.simple_dynamic_weight import *
# TODO:修改其他
__DYNAMICS__ = {
    "UAU": SimpleDynamicWeightUAU,
    "CompUAU": SimpleDynamicWeightCompUAU,
    "SCUAU": SimpleDynamicWeightSCUAU,
    "SCCompUAU": SimpleDynamicWeightSCCompUAU,
}


def get(NAME):
    if NAME in __DYNAMICS__:
        return __DYNAMICS__[NAME]
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__DYNAMICS__.keys())}"
        )
