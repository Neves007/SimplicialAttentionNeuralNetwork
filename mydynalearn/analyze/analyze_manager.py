from mydynalearn.analyze.analyzer import *
from mydynalearn.config import Config

class AnalyzeManager():
    def __init__(self,train_experiment_manager):
        config_analyze = Config.get_config_analyze()
        self.config = config_analyze['default']
        self.train_experiment_manager = train_experiment_manager
        self.r_value_analyzer = RValueAnalyzer(self.config)

    def get_model_executor_generator(self):
        ''' 模型测试执行器的生成器

        :yied: 生成所有的模型测试执行器
        '''
        exp_generator = self.train_experiment_manager.get_exp_generator() # 所有的实验对象
        for exp in exp_generator:
            # 读取数据
            network, dynamics, _, _, test_loader = exp.create_dataset()
            epoch_tasks = exp.model.epoch_tasks
            EPOCHS = epoch_tasks.EPOCHS
            for model_exp_epoch_index in range(EPOCHS):
                # 将数据集带入模型执行结果
                # 第epoch次迭代训练的模型model_exp，测试集为dataset_exp的数据集。
                model_executor = runModelOnTestData(self.config,
                                             network,
                                             dynamics,
                                             test_loader,
                                             model_exp_epoch_index,
                                             model_exp=exp,
                                             dataset_exp=exp,
                                             )
                yield model_executor

    def analyze_trained_model(self):
        '''
        将所有实验的数据集引入到自己的模型中，输出analyze_result
        '''
        # 把这个testresult
        print("*"*10+" ANALYZE TRAINED MODEL "+"*"*10)
        model_executor_generator = self.get_model_executor_generator()
        if not os.path.exists(self.r_value_analyzer.r_value_dataframe_file_path):
            for model_executor in model_executor_generator:
                analyze_result = model_executor.run()
                # 添加结果的r值
                self.r_value_analyzer.add_r_value(analyze_result)
                torch.cuda.empty_cache()
                # 保存结果
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
        self.analyze_trained_model()
        self.analyze_stable_r_value()


