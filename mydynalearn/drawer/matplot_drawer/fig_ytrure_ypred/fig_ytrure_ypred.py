import os.path

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.drawer.utils.utils import _get_metrics

import itertools
class FigYtrureYpred():
    def __init__(self,STATES_MAP):
        all_colors = ["red","orange","yellow","green","cyan","blue","purple",'brown', 'stategrey', 'pink']
        self.STATES_MAP = STATES_MAP
        self.transition_lables = self.get_transition_lables()
        self.colors = [all_colors[i] for i in range(len(self.transition_lables))]
        self.markers = ["o" for i in range(len(self.transition_lables))]
    def get_transition_lables(self):
        '''通过对动力学状态的排列组合得出，转换的的标签

        :return:
        '''
        STATES = self.STATES_MAP.keys()
        # 使用 itertools.product 生成状态转换的所有可能组合
        transitions = itertools.product(STATES, repeat=2)
        transition_lables = [f'{start}_to_{end}' for start, end in transitions]
        return transition_lables

    def get_marker_size(self,w_T,max=10,min=2,**kwargs):
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
        self.ax.legend(handles=self.legend_elements, labels=self.transition_lables)
        self.ax.grid(True)
    def save_fig(self,fig_file):
        self.fig.savefig(fig_file)
        plt.close(self.fig)
    def scatterT(self, performance_index,performance_data,model_exp_epoch_index,**kwargs):
        self.fig, self.ax = plt.subplots()
        marker_size = self.get_marker_size(max=700, min=50,**kwargs)
        for index in range(len(self.transition_lables)):
            self.ax.scatter(x=performance_data[index][:, 0].detach().numpy(),
                            y=performance_data[index][:, 1].detach().numpy(),
                            c=self.colors[index],
                            marker=self.markers[index],
                            s=marker_size[performance_index[index]], alpha=0.3)
        self.editAix(model_exp_epoch_index,performance_data)

    def get_legend_elements(self):
        legend_elements = [plt.scatter([0],
                                       [0],
                                       c=self.colors[index],
                                       marker=self.markers[index],
                                       s=53,
                                       alpha=0.8) for index in range(len(self.transition_lables))]
        return legend_elements