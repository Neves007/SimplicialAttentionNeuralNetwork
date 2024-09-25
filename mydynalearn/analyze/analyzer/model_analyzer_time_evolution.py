from mydynalearn.dataset import *
from mydynalearn.analyze.utils.data_handler import *
from mydynalearn.config import Config

from .model_analyzer import ModelAnalyzer
from mydynalearn.util.lazy_loader import PickleLazyLoader


class ModelAnalyzerTimeEvolution(PickleLazyLoader, ModelAnalyzer):
    config = Config.get_config_analyze()['default']
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        ModelAnalyzer.__init__(self,config, exp, epoch_index)
        self.data_file = self.get_data_file(type='time_evolution')
        PickleLazyLoader.__init__(self,self.data_file)

    def create_dynamic_dataset_time_evolution(self):
        """
        创建时间演化的动力学数据集
        """
        self.dynamics_dataset_time_evolution = DynamicDatasetTimeEvolution(self.exp, self.test_model, self.network,
                                                                           self.dynamics)
        return self.dynamics_dataset_time_evolution.get_data()


    def analyze_model_performance_time_evolution(self):
        """分析时间演化数据
        :return:
        """
        self.logger.increase_indent()
        self.logger.log(f"analyze time evolution model performance of epoch {self.epoch_index}")
        dynamic_dataset_time_evolution = self.create_dynamic_dataset_time_evolution()
        handler_performance_generator_time_evolution = PerformanceResultGeneratorTimeEvolutionHandler(self.exp,
                                                                                                      self.epoch_index,
                                                                                                      self.network,
                                                                                                      self.dynamics,
                                                                                                      dynamic_dataset_time_evolution)
        analyze_result_model_performance_time_evolution = handler_performance_generator_time_evolution.create_analyze_result()
        self.logger.decrease_indent()
        return analyze_result_model_performance_time_evolution

    def _create_data(self):
        self.network = self.exp.network.get_data()
        analyze_result = self.analyze_model_performance_time_evolution()
        return analyze_result