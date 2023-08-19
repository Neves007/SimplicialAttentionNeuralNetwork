import torch
from .fig import *
from mydynalearn.transformer import data_curEpoch_2_data_T
from mydynalearn.drawer.matplot_drawer.drawer_matplot import MatplotDrawer
from mydynalearn.drawer.utils import *
import pickle
import matplotlib.pyplot as plt
from mydynalearn.drawer.matplot_drawer.fig.fig_ytrure_ypred.getter import get as fig_ytrure_ypred_getter

class DrawerMatplotEpochFigYtrureYpred(MatplotDrawer):
    def __init__(self,config,dynamics):
        super().__init__(config)
        self.dynamics = dynamics
        self.epochs = config.dataset.epochs
        self.figName = "/epoch_performance_fig_ytrure_ypred.png"
        self.Fig_ytrure_ypred = fig_ytrure_ypred_getter(config)

    def save_epoch_data(self, epoch_index, testResult):
        self.fileName = self.path_to_epochData + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(self.fileName, "wb") as file:
            pickle.dump(testResult,file)
    def load_epoch_data(self, epoch_index):
        fileName = self.path_to_epochData + "/epoch{:d}Data.pkl".format(epoch_index)
        with open(fileName, "rb") as file:
            epochData = pickle.load(file)
        return epochData
    def save_fig(self):
        self.fig.tight_layout()
        fig_name = self.path_to_fig + self.figName
        self.fig.savefig(fig_name)
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
        zoomtimes=0.8
        self.fig, self.axes = plt.subplots(self.row,self.col,figsize=(zoomtimes*37, zoomtimes*35),dpi=200)

    def _get_subplotIndex(self,fig_index):
        # 通过epochindex拿到子图axis
        row_index = int(fig_index / self.col)
        col_index = fig_index % self.col
        if self.row==self.col==1:
            ax = self.axes
        elif self.row==1 and self.col>1:
            ax = self.axes[col_index]
        elif self.row>1 and self.col>1:
            ax = self.axes[row_index][col_index]
        return ax

    def draw(self, epoch_index, epochData):
        x_T, y_pred_T, y_ob_T, y_true_T,w_T = data_curEpoch_2_data_T(epochData,self.config.is_weight)

        ax = self._get_subplotIndex(epoch_index)
        fig_ytrure_ypred = self.Fig_ytrure_ypred(ax,self.dynamics)
        fig_ytrure_ypred.scatterT(epoch_index, x_T, y_pred_T, y_ob_T, y_true_T,w_T)
        fig_ytrure_ypred.editAix(epoch_index,x_T, y_pred_T, y_ob_T, y_true_T)

