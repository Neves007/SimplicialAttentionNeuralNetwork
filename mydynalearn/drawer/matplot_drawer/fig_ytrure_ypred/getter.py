from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred import *

__FIGS__ = {
    "UAU": FigYtrureYpredUAU,
    "CompUAU": FigYtrureYpredCompUAU,
    "SCUAU": FigYtrureYpredUAU,
    "SCCompUAU": FigYtrureYpredCompUAU,
}


def get(dynamics):
    NAME = dynamics.NAME
    if NAME in __FIGS__:
        return __FIGS__[NAME](dynamics)
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__FIGS__.keys())}"
        )
