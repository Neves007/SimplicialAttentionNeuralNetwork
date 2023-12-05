import torch
from .fig import *
from mydynalearn.drawer_old.matplot_drawer.drawer_matplot import MatplotDrawer
from mydynalearn.drawer_old.utils import *
import pickle
import matplotlib.pyplot as plt
from mydynalearn.drawer_old.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer_old.matplot_drawer.fig.fig_ytrure_ypred.getter import get as fig_ytrure_ypred_getter
from mydynalearn.drawer_old.utils.performance_data.getter import get as performance_data_getter


class DrawerMatplotEpochFigYtrureYpred(MatplotDrawer):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.dynamics = dynamics
        self.epochs = config.dataset.epochs
        self.figName = "/epoch_performance_fig_ytrure_ypred.png"
        self.zoomtimes = 0.8 # 图片大小
        self.Fig_ytrure_ypred = fig_ytrure_ypred_getter(config)
    def save_fig(self):
        self.fig.tight_layout()
        fig_name = self.path_to_fig + self.figName
        self.fig.savefig(fig_name)
        plt.close()
    def split_subplots(self, totalFigures):
        # 分割子图
        row = totalFigures ** 0.5
        if row % 1 ==0:
            col = row
        else:
            row = int(row)+1
            col = row
        self.totalFigures = int(totalFigures)
        self.row = int(row)
        self.col = int(col)
        self.fig, self.axes = plt.subplots(self.row,self.col,figsize=(self.zoomtimes*37, self.zoomtimes*35),dpi=200)

    def _get_subplotIndex(self,epoch_index):
        # 通过epochindex拿到子图axis
        row_index = int(epoch_index / self.col)
        col_index = epoch_index % self.col
        if self.row==self.col==1:
            ax = self.axes
        elif self.row==1 and self.col>1:
            ax = self.axes[col_index]
        elif self.row>1 and self.col>1:
            ax = self.axes[row_index][col_index]
        return ax

    def draw(self, **kwargs):
        get_performance_index,get_performance_data = performance_data_getter(self.config)
        epoch_index = kwargs["epoch_index"]
        ax = self._get_subplotIndex(epoch_index)
        dynamic_data_handler = DynamicDataHandler(**kwargs)
        performance_index = get_performance_index(dynamic_data_handler)
        performance_data = get_performance_data(dynamic_data_handler)
        fig_ytrure_ypred = self.Fig_ytrure_ypred(ax,self.dynamics)
        fig_ytrure_ypred.scatterT(performance_index,performance_data,**kwargs)
        fig_ytrure_ypred.editAix(epoch_index,performance_data)

