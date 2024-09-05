from .model_analyzer import ModelAnalyzer,BestEpochHandler
from mydynalearn.logger import Log
class ExpModelsAnalyzer:
    def __init__(self, config, exp):
        """
        初始化 ExpModelsAnalyzer
        :param exp: 实验对象
        """
        self.config = config
        self.logger = Log("ExpModelsAnalyzer")
        self.exp = exp
        self.EPOCHS = exp.config.model.EPOCHS
        self.best_epoch_index = 0
        self.model_performance_analyze_result_generator = list(self.get_model_performance_analyze_result_generator())
        self.best_epoch_handler = BestEpochHandler(self.config,self.exp, self.model_performance_analyze_result_generator)


    def get_model_performance_analyze_result(self,epoch_index):
        model_analyzer = ModelAnalyzer(self.config, self.exp, epoch_index)
        model_performance_analyze_result = model_analyzer.get_normal_performance_analysis_result()
        return model_performance_analyze_result

    def model_performance_analyze_result_is_exist(self, epoch_index):
        model_analyzer = ModelAnalyzer(self.config, self.exp, epoch_index)
        result_file_is_exist = model_analyzer.result_file_is_exist()
        return result_file_is_exist

    def get_model_performance_analyze_result_time_evolutioin(self,epoch_index):
        model_analyzer = ModelAnalyzer(self.config, self.exp, epoch_index)
        model_performance_analyze_result = model_analyzer.get_time_evolution_performance_analysis_result()
        return model_performance_analyze_result


    def get_model_performance_analyze_result_generator(self):
        for epoch_index in range(self.EPOCHS):
            yield self.get_model_performance_analyze_result(epoch_index)
    def analyze_all_epochs_model(self):
        """
        运行exp中所有epoch的模型分析结果
        :return:
        """
        for epoch_index in range(self.EPOCHS):
            if not self.model_performance_analyze_result_is_exist(epoch_index):
                model_analyzer = ModelAnalyzer(self.config, self.exp, epoch_index)
                # 只运行普通的衡量结果
                model_analyzer.run_normal_performance_analysis()
    def find_best_epoch(self):
        """
        """
        best_epoch_exp_item, best_epoch_index = self.best_epoch_handler.find_best_epoch()
        return best_epoch_exp_item, best_epoch_index


    def analyze_best_epoch_model(self):
        """
        :return:
        """
        best_epoch_exp_item, best_epoch_index = self.find_best_epoch()
        self.get_model_performance_analyze_result_time_evolutioin(best_epoch_index)




    def run(self):
        """
        对单个实验进行分析
        """
        # 对所有epoch的模型进行分析
        self.logger.increase_indent()
        self.logger.log(f"Analyze model to each epoch in experiment：{self.exp.NAME}")
        self.analyze_all_epochs_model()
        # 对best epoch的模型进行分析
        self.logger.log(f"analysis the best epoch model in experiment：{self.exp.NAME}")
        self.analyze_best_epoch_model()
        self.logger.decrease_indent()