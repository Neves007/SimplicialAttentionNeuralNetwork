from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.drawer.utils.performace_data.utils import _get_metrics
class FigYtrureYpred():
    def __init__(self,ax,dynamics):
        self.STATES_MAP = dynamics.STATES_MAP
        self.dynamics = dynamics
        self.ax = ax
        self.colors = None
        self.markers = None
        self.label = None

    def get_marker_size(self,w_T,max=10,min=2,**kwags):

        w_min = w_T.min()
        w_max = w_T.max()
        new_min = min
        new_max = max
        if w_min == w_max:
            size = min*torch.ones(w_T.shape)
        else:
            size = (w_T - w_min) / (w_max - w_min) * (new_max - new_min) + new_min
        return size.to(torch.long)

    def editAix(self,epoch_index,performance_data):
        corrcoef,r2 = _get_metrics(performance_data)
        self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}'.format(epoch_index, corrcoef))
        self.ax.set_xticks(np.linspace(0,1,5))
        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("Target")  # 设置x轴标注
        self.ax.set_ylabel("prediction")  # 设置y轴标注

        self.legend_elements = self.get_legend_elements()
        self.ax.legend(handles=self.legend_elements, labels=self.label)
        self.ax.grid(True)

    def scatterT(self, performance_index,performance_data,**kwags):
        marker_size = self.get_marker_size(max=700, min=50,**kwags)
        for index in range(len(self.label)):
            self.ax.scatter(x=performance_data[index][:, 0].detach().numpy(),
                            y=performance_data[index][:, 1].detach().numpy(),
                            c=self.colors[index],
                            marker=self.markers[index],
                            s=marker_size[performance_index[index]], alpha=0.3)

    def get_legend_elements(self):
        legend_elements = [plt.scatter([0],
                                       [0],
                                       c=self.colors[index],
                                       marker=self.markers[index],
                                       s=53,
                                       alpha=0.8) for index in range(len(self.label))]
        return legend_elements