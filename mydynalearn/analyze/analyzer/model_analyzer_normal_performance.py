from mydynalearn.dataset import *
from mydynalearn.analyze.utils.data_handler import *
from mydynalearn.config import Config
import os
import pickle
from mydynalearn.logger import Log
from mydynalearn.util.lazy_loader import PickleLazyLoader
from .model_analyzer import ModelAnalyzer

class ModelAnalyzerNormalPerformance(PickleLazyLoader, ModelAnalyzer):
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        ModelAnalyzer.__init__(self,config, exp, epoch_index)
        self.data_file = self.get_data_file(type='normal_performance')
        PickleLazyLoader.__init__(self,self.data_file)

    def analyze_model_performance(self):
        """
        :return:
        """
        dataset = self.exp.dataset.load()
        generator_performance_result = self.test_model.run_test_epoch(**dataset)
        handler_performance_generator = PerformanceResultGeneratorHandler(self.exp,
                                                                          self.epoch_index,
                                                                          generator_performance_result,**dataset)
        analyze_result = handler_performance_generator.create_analyze_result()
        return analyze_result

    def _create_data(self):
        self.network = self.exp.network.get_data()
        analyze_result = self.analyze_model_performance()
        return analyze_result