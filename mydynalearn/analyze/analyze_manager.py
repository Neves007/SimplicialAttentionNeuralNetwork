from mydynalearn.analyze.analyzer import *

class AnalyzeManager():
    def __init__(self,train_experiment_manager):

        self.config = train_experiment_manager.config
        self.train_experiment_manager = train_experiment_manager
        self.r_value_analyzer = RValueAnalyzer(self.config)


    def analyze_trained_model(self):
        '''
        将所有实验的数据集引入到自己的模型中，输出analyze_result
        '''
        # 把这个testresult
        print("*"*10+" ANALYZE TRAINED MODEL "+"*"*10)
        exp_iter = self.train_experiment_manager.get_exp_iter()

        if not os.path.exists(self.r_value_analyzer.r_value_dataframe_file_path):
            for exp in exp_iter:
                network, dynamics, train_loader, val_loader, test_loader = exp.create_dataset()
                epoch_tasks = exp.model.epoch_tasks
                EPOCHS = epoch_tasks.EPOCHS
                for model_exp_epoch_index in range(EPOCHS):
                    # 将数据集带入模型执行结果
                    model_executor = runModelOnTestData(self.config,
                                                 network,
                                                 dynamics,
                                                 test_loader,
                                                 model_exp_epoch_index,
                                                 model_exp=exp,
                                                 dataset_exp=exp,
                                                 )
                    analyze_result = model_executor.run()
                    # 添加结果的r值
                    self.r_value_analyzer.add_r_value(analyze_result)
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


