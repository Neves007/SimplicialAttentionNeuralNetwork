from mydynalearn.analyze.analyzer import *
from mydynalearn.config import Config

class AnalyzeManager():
    def __init__(self,train_experiment_manager):
        config_analyze = Config.get_config_analyze()
        self.config = config_analyze['default']
        self.train_experiment_manager = train_experiment_manager
        self.r_value_analyzer = RValueAnalyzer(self.config)

    def _get_model_executor_generator(self):
        exp_generator = self.train_experiment_manager.get_exp_generator()  # 所有的实验对象
        for exp in exp_generator:
            # 读取数据
            network, dynamics, _, _, test_loader = exp.create_dataset()
            model_executor = runModelOnTestData(self.config,
                                                network,
                                                dynamics,
                                                test_loader,
                                                model_exp=exp,
                                                dataset_exp=exp,
                                                )
            yield model_executor

    def get_analyze_result_generator_for_best_epoch(self):
        model_executor_generator = self._get_model_executor_generator()
        for model_executor in model_executor_generator:
            best_epoch = self.r_value_analyzer.get_best_epoch(**model_executor.global_info)
            best_epoch_analyze_result = model_executor.run(best_epoch)
            yield best_epoch_analyze_result
            
    def get_analyze_result_generator_for_all_epochs(self):
        model_executor_generator = self._get_model_executor_generator()
        for model_executor in model_executor_generator:
            EPOCHS = model_executor.EPOCHS
            for model_exp_epoch_index in range(EPOCHS):
                analyze_result = model_executor.run(model_exp_epoch_index)
                yield analyze_result

    def add_r_value_for_all_epochs(self):
        '''
        将所有实验的数据集引入到自己的模型中，输出analyze_result
        '''
        print("*" * 10 + " ANALYZE TRAINED MODEL " + "*" * 10)
        analyze_result_generator = self.get_analyze_result_generator_for_all_epochs()
        r_value_dataframe_is_exist = os.path.exists(self.r_value_analyzer.r_value_dataframe_file_path)
        for analyze_result in analyze_result_generator:
            if not r_value_dataframe_is_exist:
                self.r_value_analyzer.add_r_value(analyze_result)

        if not r_value_dataframe_is_exist:
            self.r_value_analyzer.save_r_value_dataframe()
        else:
            self.r_value_analyzer.load_r_value_dataframe()


    def analyze_stable_r_value(self):
        '''分析R值的稳定点

        io:  stable_r_value_dataframe.csv
        ''' 
        if not os.path.exists(self.r_value_analyzer.stable_r_value_dataframe_file_path):
            self.r_value_analyzer.analyze_stable_r_value()
        else:
            self.r_value_analyzer.load_stable_r_value_dataframe()
    



    def run(self):
        '''
        分析训练数据，为画图做准备
        输出：
        '''
        self.add_r_value_for_all_epochs()
        self.analyze_stable_r_value()


