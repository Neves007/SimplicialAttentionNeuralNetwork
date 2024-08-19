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
        self.shrink_times = 1
        pass

    def save_fig(self,fig_file_path):
        self.fig.savefig(fig_file_path)
        plt.close(self.fig)


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

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
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

    def draw(self):
        # 设置点的大小范围，例如从10到100
        size_range = (20, 500)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="trans_prob_true", y="trans_prob_pred",
                        hue="pred_trans_type",
                        palette=self.palette,
                        size="weight",  # 使用 'weight' 列来决定每个点的大小
                        sizes=size_range,  # 设置点的大小范围
                        linewidth=0,
                        alpha=0.1,
                        data=self.test_result_df, ax=self.ax)


class FigBetaRho():
    def __init__(self,dynamics, x, stady_rho_dict,**kwargs):
        self.x = x
        self.stady_rho_dict = stady_rho_dict
        self.state_list = dynamics.STATES_MAP.keys()
        self.dynamic_name = dynamics.NAME
        self.num_fig = len(self.state_list)

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
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.subplots_adjust(top=0.981, bottom=0.123, left=0.091, right=0.99, hspace=0.275, wspace=0.375)
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".2%", linewidths=.5, ax=ax,cmap="Blues")
        self.ax = ax
        self.fig = fig


    def edit_ax(self):
        self.ax.set_title("Normalized Confusion Matrix")
        self.ax.set_ylabel("Predicted lable")  # 设置x轴标注
        self.ax.set_xlabel("True lable")  # 设置y轴标注
        self.ax.grid(False)
        plt.tight_layout()



class FigActiveNeighborsTransprob(MatplotDrawer):
    def __init__(self,
                 test_result_info,
                 test_result_df,
                 dynamics_STATES_MAP,
                 model_performace_dict,):
        super().__init__()
        self.STATES_MAP = dynamics_STATES_MAP
        self.test_result_df = test_result_df
        self.shrink_times = 1.5

    def edit_ax(self):
        '''
        编辑ax
        :return:
        '''
        # self.ax.set_title(r' $R$ = {:0.5f}'.format(self.corrcoef))
        # self.ax.set_xticks(np.linspace(0,1,5))
        # 自定义图例
        from matplotlib.lines import Line2D
        # 获取唯一的 transition_type
        unique_types = self.test_result_df['transition_type'].unique()
        # 创建自定义图例项
        legend_elements = [
            Line2D([0], [0], color=sns.color_palette("tab10")[i], lw=2, linestyle='-', label=f"{ut} (True)")
            for i, ut in enumerate(unique_types)]
        legend_elements += [
            Line2D([0], [0], color=sns.color_palette("tab10")[i], lw=2, linestyle='--', label=f"{ut} (Pred)")
            for i, ut in enumerate(unique_types)]

        # 添加自定义图例到图形
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Transition Type")

        self.ax.set_yticks(np.linspace(0,1,5))
        self.ax.set_xlim(left=0)
        self.ax.set_ylim([0,1])
        self.ax.set_xlabel("k")  # 设置x轴标注
        self.ax.set_ylabel("State transition probability")  # 设置y轴标注
        self.ax.grid(True)
        # 显示图形
        plt.tight_layout()
        plt.show()


    def draw(self):
        '''
        绘制图像
        :return:
        '''
        self.fig, self.ax = plt.subplots(figsize=(10/self.shrink_times, 6/self.shrink_times))
        # 创建一个新的列来表示节点状态迁移的类型
        self.test_result_df['transition_type'] = self.test_result_df.apply(
            lambda row: f"{row['x_lable']} → {row['y_ob_lable']}", axis=1
        )
        # 绘制 trans_prob_true 的实线图（真实数据）
        sns.lineplot(
            x='adj_act_edges',
            y='trans_prob_true',
            hue='transition_type',
            data=self.test_result_df,
            ax=self.ax,
            palette="tab10",  # 使用默认的10种颜色调色板
            linestyle="-",  # 实线表示真实值
            linewidth=3,  # 设置线条粗细为 2
            alpha=0.8,  # 设置透明度为 0.8
            legend=False  # 暂时不显示图例，后面一起处理
        )

        # 绘制 trans_prob_pred 的虚线图（预测数据）
        sns.lineplot(
            x='adj_act_edges',
            y='trans_prob_pred',
            hue='transition_type',
            data=self.test_result_df,
            ax=self.ax,
            palette="tab10",  # 使用相同的调色板
            linestyle="--",  # 虚线表示预测值
            linewidth=3,  # 设置线条粗细为 2
            alpha=0.8,  # 设置透明度为 0.8
            legend=False  # 暂时不显示图例，后面一起处理
        )




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
