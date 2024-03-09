import os.path

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.drawer.utils.utils import _get_metrics

from mydynalearn.analyze.utils.data_handler.dynamic_data_handler import DynamicDataHandler
import itertools

class FigYtrureYpred():
    def __init__(self, model_dynamics,  model_exp_epoch_index,  test_result_time_list, **kwargs):
        colormap_dict = {
            "UAU": 'viridis',
            "CompUAU": 'viridis',
            "CoopUAU": 'viridis',
            "AsymUAU": 'viridis',
            "SCUAU": 'plasma',
            "SCCompUAU": 'plasma',
            "SCCoopUAU": 'plasma',
            "SCAsymUAU": 'plasma',
        }


        self.dynamics = model_dynamics
        self.epoch_index = model_exp_epoch_index
        self.dynamic_data_handler = DynamicDataHandler(model_dynamics, test_result_time_list)
        self.performance_index = self.dynamic_data_handler.get_performance_index()
        self.performance_data = self.dynamic_data_handler.get_performance_data()

        self.STATES_MAP = model_dynamics.STATES_MAP
        self.transition_lables = self.dynamic_data_handler.get_transition_lables()
        # self.colors = [all_colors[i] for i in range(len(self.transition_lables))]

        cmap = plt.cm.get_cmap(colormap_dict[model_dynamics.NAME], len(self.transition_lables))

        # 生成10个等间隔的值
        indices = np.linspace(0, 1, len(self.transition_lables))

        # 提取颜色

        # 将颜色转换为RGB格式
        self.colors = [color[:3] for color in cmap(indices)]
        self.markers = ["o" for i in range(len(self.transition_lables))]



    def get_marker_size(self):
        max = 700
        min = 50
        w_min = self.dynamic_data_handler.w_T.min()
        w_max = self.dynamic_data_handler.w_T.max()
        new_min = min
        new_max = max
        if w_min == w_max:
            size = min*torch.ones(self.dynamic_data_handler.w_T.shape)
        else:
            size = (self.dynamic_data_handler.w_T - w_min) / (w_max - w_min) * (new_max - new_min) + new_min
        return size.to(torch.long)

    def editAix(self):
        corrcoef,r2 = _get_metrics(self.performance_data)
        self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}'.format(self.epoch_index, corrcoef))
        self.ax.set_xticks(np.linspace(0,1,5))
        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("Target")  # 设置x轴标注
        self.ax.set_ylabel("prediction")  # 设置y轴标注

        self.legend_elements = self.get_legend_elements()
        self.ax.legend(handles=self.legend_elements, labels=self.transition_lables, loc='upper left'              )
        self.ax.grid(True)
    def save_fig(self,fig_file):
        self.fig.savefig(fig_file)
        plt.close(self.fig)
    def scatterT(self):
        self.fig, self.ax = plt.subplots()

        marker_size = self.get_marker_size()
        for index in range(len(self.transition_lables)):
            self.ax.scatter(x=self.performance_data[index][:, 0].detach().numpy(),
                            y=self.performance_data[index][:, 1].detach().numpy(),
                            c=np.array([self.colors[index]]),
                            marker=self.markers[index],
                            s=marker_size[self.performance_index[index]], alpha=0.3)
        self.editAix()

    def get_legend_elements(self):
        legend_elements = [plt.scatter([0],
                                       [0],
                                       c=np.array([self.colors[index]]),
                                       marker=self.markers[index],
                                       s=53,
                                       alpha=0.8) for index in range(len(self.transition_lables))]
        return legend_elements