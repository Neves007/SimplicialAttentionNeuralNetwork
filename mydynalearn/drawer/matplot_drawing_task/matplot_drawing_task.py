import os
from mydynalearn.config import *
from mydynalearn.drawer.MatplotDrawer.matplot_drawer import *
from mydynalearn.analyze.analyzer import runModelOnTestData, epochAnalyzer
import pandas as pd
from mydynalearn.analyze.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer.utils.utils import _get_metrics
from mydynalearn.experiments.train_experiment_manager import TrainExperimentManager
import re
import itertools


class MatplotDrawingTask():
    '''
    Matplot图片绘制任务
    - 准备绘图数据
    - 使用数据绘制图像
    - 保存图像
    '''

    def __init__(self, analyze_manager):
        self.analyze_manager = analyze_manager
        self.config_drawer = Config.get_config_drawer()
        self.best_epoch_dataframe = self.analyze_manager.epoch_analyzer.best_epoch_dataframe

    def _get_analyze_result(self, data):
        '''
        输入一行self.best_epoch_dataframe数据，返回对应的分析结果
        :param data: 一行数据记录
        :return:
        '''
        model_exp_epoch_index = int(data['model_epoch_index'])
        network = data['model_network_name']
        dynamics = data['model_dynamics_name']
        model = data['model_name']
        exp = self.analyze_manager.train_experiment_manager.get_train_exp(network, dynamics, model)
        model_executor = runModelOnTestData(self.analyze_manager.config,
                                            model_exp=exp,
                                            dataset_exp=exp,
                                            )
        analyze_result = model_executor.run(model_exp_epoch_index)
        return analyze_result

    def _make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _save_fig(self, fig_drawer, test_result_info):
        fig_file_path = test_result_info['fig_file_path']
        fig_drawer.save_fig(fig_file_path)
        print("draw " + fig_file_path)

    def _get_drawing_data_generator(self):
        '''
        准备绘图数据
        :return:
        '''
        # 拿到最佳结果的数据
        best_epoch_dataframe = self.best_epoch_dataframe
        for index, data_info in best_epoch_dataframe.iterrows():
            # 通过数据信息拿到数据结果
            data_result = self._get_analyze_result(data_info)
            # 拿到图片名字
            fig_file_path = self.get_fig_file_path(data_info)
            # 组成绘图数据
            drawing_data = {
                "data_info": data_info,
                "data_result": data_result,
                "fig_file_path": fig_file_path,
            }
            yield drawing_data

    def run(self):
        # 绘图数据的generator
        drawing_data_generator = self._get_drawing_data_generator()
        for drawing_data in drawing_data_generator:
            # 遍历每一个数据进行绘制
            fig_drawer = self.draw_fig(drawing_data)
            # 图片保存
            self._save_fig(fig_drawer, drawing_data)

    def draw_fig(self, drawing_data):
        pass


class FigYtrureYpredDrawingTask(MatplotDrawingTask):
    def __init__(self, analyze_manager):
        super().__init__(analyze_manager)
        self.config = self.config_drawer['fig_ytrure_ypred']

    def get_fig_file_path(self, data_info):
        '''
        获取图片的储存路径
        :param data_info: 数据的基本信息
        :return: fig_file_path
        '''
        # 初始化数据
        network_name = data_info["model_network_name"]
        dynamics_name = data_info["model_dynamics_name"]
        model_name = data_info["model_name"]
        epoch_index = data_info['model_epoch_index']
        # 获得模型整体信息
        model_info = str.join('_', [network_name, dynamics_name, model_name])
        # 图片名称
        fig_name = "FigYtrureYpred_{}_epoch_{}".format(model_info, epoch_index)
        # 图片所在目录
        fig_dir_root_path = self.config.fig_dir_path
        fig_dir_path = os.path.join(fig_dir_root_path, network_name)
        self._make_dir(fig_dir_path)
        fig_file_path = os.path.join(fig_dir_path, fig_name)
        return fig_file_path

    def draw_fig(self, drawing_data):
        '''
        绘制图片draw_fig
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        print("*" * 10 + "DRAW FigYtrureYpred" + "*" * 10)
        # 图像绘制
        fig_drawer = FigYtrureYpred(**data_result)
        fig_drawer.draw()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer


class FigBetaRhoDrawingTask(MatplotDrawingTask):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.config = self.config_drawer['fig_beta_rho']


class FigConfusionMatrixDrawingTask(FigYtrureYpredDrawingTask):
    '''
    混淆矩阵表格
    '''

    def __init__(self, analyze_manager):
        # todo: 绘制混淆矩阵
        super(FigConfusionMatrixDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_confusion_matrix']

    def get_fig_file_path(self, data_info):
        '''
        获取图片的储存路径
        :param data_info: 数据的基本信息
        :return: fig_file_path
        '''
        # 初始化数据
        network_name = data_info["model_network_name"]
        dynamics_name = data_info["model_dynamics_name"]
        model_name = data_info["model_name"]
        epoch_index = data_info['model_epoch_index']
        # 获得模型整体信息
        model_info = str.join('_', [network_name, dynamics_name, model_name])
        # 图片名称
        fig_name = "ConfusionMatrix_{}_epoch_{}".format(model_info, epoch_index)
        # 图片所在目录
        fig_dir_root_path = self.config.fig_dir_path
        fig_dir_path = os.path.join(fig_dir_root_path, network_name)
        self._make_dir(fig_dir_path)
        fig_file_path = os.path.join(fig_dir_path, fig_name)
        return fig_file_path

    def draw_fig(self, drawing_data):
        '''
        绘制图片draw_fig
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        print("*" * 10 + "DRAW FigConfusionMatrix" + "*" * 10)
        # 图像绘制
        fig_drawer = FigConfusionMatrix(**data_result)
        fig_drawer.draw()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer


class FigActiveNeighborsTransProbDrawingTask(FigYtrureYpredDrawingTask):
    '''
    激活态邻居数量-迁移概率图
    '''

    def __init__(self, analyze_manager):
        super(FigActiveNeighborsTransProbDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_active_neighbors_transprob']

    def _get_drawing_data_generator(self):
        '''
        准备绘图数据
        :yield: 一个模型训练的数据，用于画图
        '''
        # 设置配置
        best_epoch_dataframe = self.best_epoch_dataframe
        model_network_name = "ER"
        model_dynamics_name = "UAU"
        dataset_network_name = "ER"
        dataset_dynamics_name = "UAU"
        # 拿到详细信息
        best_epoch_dataframe_ER_UAU = best_epoch_dataframe[
            (best_epoch_dataframe['model_network_name'] == model_network_name) &
            (best_epoch_dataframe['model_dynamics_name'] == model_dynamics_name) &
            (best_epoch_dataframe['dataset_network_name'] == dataset_network_name) &
            (best_epoch_dataframe['dataset_dynamics_name'] == dataset_dynamics_name)
            ]
        # 拿到数据
        for index, data_info in best_epoch_dataframe_ER_UAU.iterrows():
            # 通过数据信息拿到数据结果
            data_result = self._get_analyze_result(data_info)
            # 拿到图片名字
            fig_file_path = self.get_fig_file_path(data_info)
            # 组成绘图数据
            drawing_data = {
                "data_info": data_info,
                "data_result": data_result,
                "fig_file_path": fig_file_path,
            }
            yield drawing_data

    def get_fig_file_path(self, data_info):
        '''
        获取图片的储存路径
        :param data_info: 数据的基本信息
        :return: fig_file_path
        '''
        # 初始化数据
        network_name = data_info["model_network_name"]
        dynamics_name = data_info["model_dynamics_name"]
        model_name = data_info["model_name"]
        epoch_index = data_info['model_epoch_index']
        # 获得模型整体信息
        model_info = str.join('_', [network_name, dynamics_name, model_name])
        # 图片名称
        fig_name = "FigActiveNeighborsTransProb_{}_epoch_{}".format(model_info, epoch_index)
        # 图片所在目录
        fig_dir_root_path = self.config.fig_dir_path
        fig_dir_path = os.path.join(fig_dir_root_path, network_name)
        self._make_dir(fig_dir_path)
        fig_file_path = os.path.join(fig_dir_path, fig_name)
        return fig_file_path

    def draw_fig(self, drawing_data):
        '''
        绘制图片draw_fig
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        print("*" * 10 + "DRAW FigActiveNeighborsTrans" + "*" * 10)
        # 图像绘制
        fig_drawer = FigActiveNeighborsTransprob(**data_result)
        fig_drawer.draw()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer

class FigKLossDrawingTask(FigYtrureYpredDrawingTask):
    '''
    邻居数量-loss图
    '''

    def __init__(self, analyze_manager):
        super(FigKLossDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_k_loss']

    def get_fig_file_path(self, data_info):
        pass


class FigKDistributionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    网络度分布图
    X：一阶度
    Y：如果是高阶网络，则为二阶度
    '''

    def __init__(self, analyze_manager):
        super(FigKDistributionDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_k_distribution']

    def get_fig_file_path(self, data_info):
        pass


class FigTimeEvolutionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    时间演化图
    '''

    def __init__(self, analyze_manager):
        super(FigTimeEvolutionDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_time_evolution']

    def get_fig_file_path(self, data_info):
        pass
