from mydynalearn.drawer.matplot_drawer.fig.fig_ytrure_ypred import *

__FIGS__ = {
    "UAU": FigYtrureYpredUAU,
    "CompUAU": FigYtrureYpredCompUAU,
    "SCUAU": FigYtrureYpredUAU,
    "SCCompUAU": FigYtrureYpredCompUAU,
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __FIGS__:
        return __FIGS__[NAME]
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__FIGS__.keys())}"
        )
