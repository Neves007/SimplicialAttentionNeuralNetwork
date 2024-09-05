from mydynalearn.dataset import *
from .utils.data_handler import *
from mydynalearn.config import Config
import os
import pickle
from mydynalearn.logger import Log

class ModelAnalyzer:
    config = Config.get_config_analyze()['default']
    def __init__(self, config, exp, epoch_index):
        """
        初始化 ModelAnalyzer
        :param exp: 实验对象
        """
        self.exp = exp
        self.logger = Log("ModelAnalyzer")
        self.dynamics = self.exp.dynamics
        self.epoch_index = epoch_index
        self.test_model = self._init_test_model()



    def get_analyze_result_dir_path(self):
        """
        构建分析结果的文件路径
        """
        model_info = f"{self.exp.exp_info['model_network_name']}_{self.exp.exp_info['model_dynamics_name']}_{self.exp.exp_info['model_name']}"
        testdata_info = f"{self.exp.exp_info['dataset_network_name']}_{self.exp.exp_info['dataset_dynamics_name']}_{self.exp.exp_info['model_name']}"
        model_dir_name = f"model_{model_info}"
        testdata_dir_name = f"testdata_{testdata_info}"

        analyze_result_dir_path = os.path.join(
            self.config.root_dir_path,
            self.config.analyze_result_dir_name,
            model_dir_name,
            testdata_dir_name
        )
        os.makedirs(analyze_result_dir_path, exist_ok=True)
        return analyze_result_dir_path

    def _init_test_model(self):
        test_model = self.exp.model.epoch_tasks
        test_model.load(self.epoch_index)
        return test_model

    def create_dynamic_dataset_time_evolution(self):
        """
        创建时间演化的动力学数据集
        """
        self.dynamics_dataset_time_evolution = DynamicDatasetTimeEvolution(self.exp, self.test_model, self.network,
                                                                           self.dynamics)
        return self.dynamics_dataset_time_evolution.run()

    def analyze_model_performance(self):
        """
        :return:
        """
        self.logger.increase_indent()
        self.logger.log(f"analyze model performance of epoch {self.epoch_index}")
        dataset = self.exp.dataset.load()
        generator_performance_result = self.test_model.run_test_epoch(**dataset)
        handler_performance_generator = PerformanceResultGeneratorHandler(self.exp,
                                                                          self.epoch_index,
                                                                          generator_performance_result,**dataset)
        analyze_result = handler_performance_generator.create_analyze_result()
        self.logger.decrease_indent()
        return analyze_result

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

    def result_file_is_exist(self, type='normal_performance'):
        analyze_result_dir_path = self.get_analyze_result_dir_path()
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{self.epoch_index}_{type}_analyze_result.pkl")
        return os.path.exists(analyze_result_filepath)

    def save(self, result, type):
        '''
        保存文件
        :param result:
        :param type:
        :return:
        '''
        analyze_result_dir_path = self.get_analyze_result_dir_path()
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{self.epoch_index}_{type}_analyze_result.pkl")
        with open(analyze_result_filepath, "wb") as f:
            pickle.dump(result, f)

    def load(self, type):
        """
        从保存的文件加载分析结果
        :param type: 要加载的分析类型 (例如 "normal_performance" 或 "time_evolution")
        :return: 加载的分析结果对象
        """
        analyze_result_dir_path = self.get_analyze_result_dir_path()
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{self.epoch_index}_{type}_analyze_result.pkl")
        if not os.path.exists(analyze_result_filepath):
            raise FileNotFoundError(f"No such file: {analyze_result_filepath}")

        with open(analyze_result_filepath, "rb") as f:
            result = pickle.load(f)
        return result
    @staticmethod
    def load_from_dataframe_item(dataframe_item, type="normal_performance"):
        config = ModelAnalyzer.config
        epoch_index = dataframe_item['model_epoch_index']
        model_info = f"{dataframe_item['model_network_name']}_{dataframe_item['model_dynamics_name']}_{dataframe_item['model_name']}"
        testdata_info = f"{dataframe_item['dataset_network_name']}_{dataframe_item['dataset_dynamics_name']}_{dataframe_item['model_name']}"
        model_dir_name = f"model_{model_info}"
        testdata_dir_name = f"testdata_{testdata_info}"

        analyze_result_dir_path = os.path.join(
            config.root_dir_path,
            config.analyze_result_dir_name,
            model_dir_name,
            testdata_dir_name
        )
        analyze_result_filepath = os.path.join(analyze_result_dir_path,
                                               f"epoch{epoch_index}_{type}_analyze_result.pkl")
        if not os.path.exists(analyze_result_filepath):
            raise FileNotFoundError(f"No such file: {analyze_result_filepath}")
        with open(analyze_result_filepath, "rb") as f:
            result = pickle.load(f)
        return result

    def get_normal_performance_analysis_result(self):
        try:
            # 尝试加载分析结果
            model_performance_analyze_result = self.load(type="normal_performance")
        except FileNotFoundError:
            # 如果文件不存在，则运行分析并保存结果
            self.network = self.exp.network.create_net()
            model_performance_analyze_result = self.analyze_model_performance()
            self.save(model_performance_analyze_result, type="normal_performance")
        return model_performance_analyze_result

    def get_time_evolution_performance_analysis_result(self):
        """
        对单个实验进行模型性能分析（时间演化数据集）
        """
        try:
            # 尝试加载分析结果
            model_performance_analyze_result_time_evolution = self.load(type="time_evolution")
        except FileNotFoundError:
            # 如果文件不存在，则运行分析并保存结果
            self.network = self.exp.network.create_net()
            model_performance_analyze_result_time_evolution = self.analyze_model_performance_time_evolution()
            self.save(model_performance_analyze_result_time_evolution, type="time_evolution")
        return model_performance_analyze_result_time_evolution

    def run_normal_performance_analysis(self):
        """
        对单个实验进行模型性能分析
        """
        # 只运行没有结果文件的epoch的模型
        self.logger.increase_indent()
        self.logger.log(f"analyze normal performance")
        if not self.result_file_is_exist(type="normal_performance"):
            self.logger.increase_indent()
            self.logger.log(f"analyze the model of epoch {self.epoch_index}")
            self.get_normal_performance_analysis_result()
            self.logger.decrease_indent()
        self.logger.decrease_indent()


    def run_time_evolution_performance_analysis(self):
        """
        对单个实验进行实验演化分析
        """
        # 只运行没有结果文件的epoch的模型
        self.logger.increase_indent()
        self.logger.log(f"time_evolution performance")
        if not self.result_file_is_exist(type="time_evolution"):
            self.logger.increase_indent()
            self.logger.log(f"analyze the model of epoch {self.epoch_index}")
            self.get_time_evolution_performance_analysis_result()
            self.logger.decrease_indent()
        self.logger.decrease_indent()