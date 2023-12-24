import torch
from .fig import *
from mydynalearn.drawer_old.matplot_drawer.drawer_matplot import MatplotDrawer
from mydynalearn.drawer_old.utils import *
import pickle
import matplotlib.pyplot as plt
from mydynalearn.drawer_old.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer_old.matplot_drawer.fig.fig_maxR import FigMaxR
from mydynalearn.drawer_old.utils.performance_data.getter import get as performance_data_getter
from mydynalearn.drawer_old.utils.data_handler import EpochDataHandler

class DrawerMatplotMaxR(MatplotDrawer):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.dynamics = dynamics
        self.figName = "/maxR.png"
        self.datapath_to_maxR = config.datapath_to_maxR
        self.zoomtimes = 0.8 # 图片大小
        self.epoch_data_handler = EpochDataHandler(config,dynamics)
        self.fig_maxR = FigMaxR(config,dynamics)


    def create_fig(self):
        self.fig, self.axes = plt.subplots(figsize=(0.2*37, 0.2*35),dpi=200)

    def save_fig(self):
        self.fig.tight_layout()
        fig_name = self.config.figpath_to_max_R + self.figName
        self.fig.savefig(fig_name)
        plt.close()


    def draw(self):
        ax = self.axes
        max_R = self.epoch_data_handler.load_max_R()
        print("exp name:{:s}".format(self.config.NAME))
        print("IS_WEIGHT:{}".format(self.config.IS_WEIGHT))
        print("max R: index ({:d}), value({:f})".format(max_R[0],max_R[1]))
        self.fig_maxR.scatter(ax,max_R)

