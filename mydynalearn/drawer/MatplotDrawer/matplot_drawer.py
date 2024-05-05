import os.path

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from mydynalearn.drawer.utils.utils import _get_metrics
import seaborn as sns
import pandas as pd
import re

import itertools

class MatplotDrawer():
    def __init__(self):
        pass




class FigYtrureYpred(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,
                 ):

        super().__init__()
        palette_list = {
            "UAU": 'viridis',
            "CompUAU": 'viridis',
            "CoopUAU": 'viridis',
            "AsymUAU": 'viridis',
            "SCUAU": 'plasma',
            "SCCompUAU": 'plasma',
            "SCCoopUAU": 'plasma',
            "SCAsymUAU": 'plasma',
        }
        self.palette = palette_list[test_result_info['model_dynamics_name']]
        self.epoch_index = test_result_info['model_epoch_index']
        self.corrcoef = model_performace_dict['R']
        self.STATES_MAP = dynamics_STATES_MAP
        self.test_result_df = test_result_df



    def editAix(self):
        self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef))
        # self.ax.set_title(r'epoch = {:d}, $R$ = {:0.5f}'.format(self.epoch_index, self.corrcoef))
        self.ax.set_xticks(np.linspace(0,1,5))
        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("Target")  # 设置x轴标注
        self.ax.set_ylabel("prediction")  # 设置y轴标注
        self.ax.legend(title="Transition type", loc='upper left')
        self.ax.grid(True)
    def save_fig(self,fig_file):
        self.fig.savefig(fig_file)
        plt.close(self.fig)
    def draw(self):
        self.fig, self.ax = plt.subplots()
        # todo：修改主题
        sns.scatterplot(x="trans_prob_true", y="trans_prob_pred",
                        hue="trans_type",
                        palette=self.palette,
                        sizes=8,
                        linewidth=0,
                        alpha=0.5,
                        data=self.test_result_df, ax=self.ax)


class FigBetaRho():
    def __init__(self,dynamics, x, stady_rho_dict,**kwargs):
        self.x = x
        self.stady_rho_dict = stady_rho_dict
        self.state_list = dynamics.STATES_MAP.keys()
        self.dynamic_name = dynamics.NAME
        self.num_fig = len(self.state_list)

    def save_fig(self,fig_file):
        self.fig.savefig(fig_file)
        plt.close(self.fig)
    def draw(self):
        self.fig, ax = plt.subplots(1, self.num_fig, figsize=(5 * self.num_fig, 4.5))
        for i, state in enumerate(self.state_list):
            y = self.stady_rho_dict[state]
            ax[i].set_title('{} dynamic model'.format(self.dynamic_name))
            ax[i].plot(self.x, y,marker='^')  # Plot x vs. y with circle markers
            ax[i].set_xlabel("Effective Infection Rate")  # X-axis label
            ax[i].set_ylabel("$\\rho_{{{:s}}}$".format(state))  # Y-axis label
            ax[i].set_ylim([0, 1])  # Set the limits for the y-axis
            ax[i].set_xlim([0, self.x[-1]])  # Set the limits for the y-axis
            ax[i].grid(True)  # Show grid
        plt.tight_layout()


class FigConfusionMatrix(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super(FigConfusionMatrix, self).__init__()
        self.confusion_matrix = model_performace_dict['cm']
        self.STATES_MAP = dynamics_STATES_MAP

    def draw(self):
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".2%", linewidths=.5, ax=ax,cmap="Blues")
        plt.show()



class FigActiveNeighborsTransprob(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super(FigActiveNeighborsTransprob, self).__init__()


class FigKLoss(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super(FigKLoss, self).__init__()


class FigKDistribution(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super(FigKDistribution, self).__init__()


class FigTimeEvolution(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super(FigTimeEvolution, self).__init__()
