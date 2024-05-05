import os
from mydynalearn.config import *
from mydynalearn.drawer.MatplotDrawer.matplot_drawer import *
from mydynalearn.analyze.analyzer import runModelOnTestData,epochAnalyzer
import pandas as pd
from mydynalearn.analyze.utils.data_handler.dynamic_data_handler import DynamicDataHandler
from mydynalearn.drawer.utils.utils import _get_metrics
from mydynalearn.experiments.train_experiment_manager import TrainExperimentManager
import re
import itertools

class MatplotDrawingTask():
    def __init__(self):
        self.config_drawer = Config.get_config_drawer()
    def make_dir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)


class FigYtrureYpredDrawingTask(MatplotDrawingTask):
    def __init__(self,analyze_manager):
        super().__init__()
        self.config = self.config_drawer['fig_ytrure_ypred']
        self.analyze_manager = analyze_manager
        self.best_epoch_dataframe = self.analyze_manager.epoch_analyzer.best_epoch_dataframe


    def get_analyze_result(self, data):
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

    def run(self):
        '''
        准备绘图数据
        :return:
        '''

        # 按照网络进行分组画图
        grouped = self.best_epoch_dataframe.groupby(["model_network_name"])
        for group_name, group_data in grouped:
            # 动力学和模型的组合
            dynamic_name_list = group_data['model_dynamics_name'].unique()
            model_name_list = group_data['model_name'].unique()
            merge_dynamic_model = itertools.product(dynamic_name_list,model_name_list)
            for dynamic_model_pair in merge_dynamic_model:
                # 拿出analyze_result数据
                dynamic_name, model_name = dynamic_model_pair
                condition = (group_data['model_dynamics_name'] == dynamic_name) & (
                            group_data['model_name'] == model_name)
                data_one_fig = group_data[condition].iloc[0]
                analyze_result = self.get_analyze_result(data_one_fig)
                # 绘制
                self.draw(analyze_result)


    def draw(self,analyze_result):
        print("*"*10+"DRAW TRAIN PERFORMANCE"+"*"*10)
        fig_drawer = FigYtrureYpred(**analyze_result)
        fig_drawer.draw()
        fig_drawer.editAix()
        self.save_fig(fig_drawer,analyze_result['test_result_info'])
        print("PROCESS COMPLETED!\n\n")

    def save_fig(self,fig_drawer,test_result_info):
        fig_file_name = self._get_fig_name(test_result_info)
        fig_drawer.save_fig(fig_file_name)
        print("draw " + fig_file_name)

    def _get_fig_name(self, analyze_result):
        model_info = str.join('_',[analyze_result["model_network_name"],
                                   analyze_result["model_dynamics_name"],
                                   analyze_result["model_name"]
                                   ])
        testdata_info = str.join('_',[analyze_result["dataset_network_name"],
                                      analyze_result["dataset_dynamics_name"],
                                      analyze_result["model_name"]
                                      ])
        fig_name = "MODEL_{}_DATA_{}_epoch_{}".format(model_info,testdata_info,analyze_result['model_epoch_index'])
        fig_dir = self.config.model_performance_figdir
        model_performance_figdir = os.path.join(fig_dir, analyze_result['model_name'])
        self.make_dir(model_performance_figdir)
        fig_file_path = os.path.join(model_performance_figdir,fig_name)
        return fig_file_path

class FigBetaRhoDrawingTask(MatplotDrawingTask):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        self.config = self.config_drawer['fig_beta_rho']


    def get_drawing_kwargs(self):
        dataset_info = self.dataset.get_info()
        fig_name = "_".join([dataset_info['network_name'], dataset_info['dynamic_name'], '.jpg'])
        data = self.dataset.get_draw_data()
        drawing_kwargs={
            "fig_name": fig_name,
            "dynamics": self.dataset.dynamics,
            "x": data['x'],
            "stady_rho_dict": data['stady_rho_dict'],
        }
        return drawing_kwargs

    def run(self):
        drawing_kwargs = self.get_drawing_kwargs()
        fig_drawer = FigBetaRho(**drawing_kwargs)
        fig_drawer.draw()
        self.save_fig(fig_drawer,**drawing_kwargs)

    def save_fig(self,fig_drawer,fig_name,**kwargs):
        fig_dir_path = self.config.fig_dir_path
        fig_file_path = os.path.join(fig_dir_path, fig_name)
        fig_drawer.save_fig(fig_file_path)


class FigConfusionMatrixDrawingTask(FigYtrureYpredDrawingTask):
    '''
    混淆矩阵表格
    '''
    def __init__(self,analyze_manager):
        super(FigConfusionMatrixDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_confusion_matrix']
    def _get_fig_name(self, analyze_result):
        pass

    def run(self):
        '''
        准备绘图数据
        :return:
        '''

        # 按照网络进行分组画图
        grouped = self.best_epoch_dataframe.groupby(["model_network_name"])
        for group_name, group_data in grouped:
            # 动力学和模型的组合
            dynamic_name_list = group_data['model_dynamics_name'].unique()
            model_name_list = group_data['model_name'].unique()
            merge_dynamic_model = itertools.product(dynamic_name_list,model_name_list)
            for dynamic_model_pair in merge_dynamic_model:
                # 拿出analyze_result数据
                dynamic_name, model_name = dynamic_model_pair
                condition = (group_data['model_dynamics_name'] == dynamic_name) & (
                            group_data['model_name'] == model_name)
                data_one_fig = group_data[condition].iloc[0]
                analyze_result = self.get_analyze_result(data_one_fig)
                # 绘制
                self.draw(analyze_result)
    def draw(self,analyze_result):
        print("*"*10+"DRAW TRAIN PERFORMANCE"+"*"*10)
        fig_drawer = FigConfusionMatrix(**analyze_result)
        fig_drawer.draw()
        fig_drawer.editAix()
        self.save_fig(fig_drawer,analyze_result['test_result_info'])
        print("PROCESS COMPLETED!\n\n")

class FigActiveNeighborsTransProbDrawingTask(FigYtrureYpredDrawingTask):
    '''
    激活态邻居数量-迁移概率图
    '''
    def __init__(self,analyze_manager):
        super(FigActiveNeighborsTransProbDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_active_neighbors_transprob']
    def _get_fig_name(self, analyze_result):
        pass


class FigKLossDrawingTask(FigYtrureYpredDrawingTask):
    '''
    邻居数量-loss图
    '''
    def __init__(self,analyze_manager):
        super(FigKLossDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_k_loss']
    def _get_fig_name(self, analyze_result):
        pass


class FigKDistributionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    网络度分布图
    '''
    def __init__(self,analyze_manager):
        super(FigKDistributionDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_k_distribution']
    def _get_fig_name(self, analyze_result):
        pass


class FigTimeEvolutionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    时间演化图
    '''
    def __init__(self,analyze_manager):
        super(FigTimeEvolutionDrawingTask, self).__init__(analyze_manager)
        self.config = self.config_drawer['fig_time_evolution']
    def _get_fig_name(self, analyze_result):
        pass

