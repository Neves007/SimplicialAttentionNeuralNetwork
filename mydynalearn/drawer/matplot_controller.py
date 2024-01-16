import torch
from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred.getter import get as performance_drawer_getter
from mydynalearn.analyze.analyzer import *
import os
class MatplotController():
    def __init__(self,analyze_manager):
        self.config = DrawerConfig().default()
        self.analyze_manager = analyze_manager
        self.TASKS = [
            "trained_model_draw",
        ]
    def run(self):
        tasks = self.TASKS

        for t in tasks:
            if t in self.TASKS:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.TASKS}`"
                )
    def make_dir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    def draw_performance(self,analyze_result,dynamics):
        fig_drawer = performance_drawer_getter(dynamics)
        fig_drawer.scatterT(**analyze_result)
        return fig_drawer
    def get_model_performance_fig_name(self,result):
        model_exp_info_dict = {
            'model_network': result['model_network'],
            'model_dynamics': result['model_dynamics'],
            'model': result['model'],
        }
        testset_exp_info_dict = {
            'dataset_network': result['dataset_network'],
            'dataset_dynamics': result['dataset_dynamics'],
        }
        model_info = "_".join([value for value in model_exp_info_dict.values()])
        testdata_info = "_".join([value for value in testset_exp_info_dict.values()])


        fig_name = "MODEL_{}_DATA_{}_epoch_{}".format(model_info,testdata_info,result['model_exp_epoch_index'])
        fig_dir = self.config.model_performance_figdir
        model_performance_figdir = os.path.join(fig_dir, result['model'])
        self.make_dir(model_performance_figdir)
        fig_file_path = os.path.join(model_performance_figdir,fig_name)
        return fig_file_path
    def get_realnet_performance_fig_name(self,real_exp,train_param,epoch_index):
        fig_dir = self.config.realnet_performance_figdir
        merged_string = '_'.join((str(value) for value in train_param))
        test_result_path = os.path.join(fig_dir, real_exp.NAME,merged_string)
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)
        file_path = os.path.join(test_result_path, merged_string+"_epoch{:d}_test_result.png".format(epoch_index))
        return file_path
    def trained_model_draw(self):
        print("*"*10+"DRAW TRAIN PERFORMANCE"+"*"*10)
        stable_r_value_dataframe = self.analyze_manager.r_value_analyzer.stable_r_value_dataframe
        for index, row in stable_r_value_dataframe.iterrows():
            param = (row['model_network'],row['model_dynamics'],row['model'],False)
            exp = self.analyze_manager.train_experiment_manager.get_train_exp(*param)
            model_exp = exp
            dataset_exp = exp
            network, dynamics, train_loader, val_loader, test_loader = exp.create_dataset()
            epoch_tasks = exp.model.epoch_tasks
            EPOCHS = epoch_tasks.EPOCHS
            model_exp_epoch_index = list(range(EPOCHS))[row['max_R_epoch']]
            model_executor = runModelOnTestData(self.analyze_manager.config,
                                                network,
                                                dynamics,
                                                test_loader,
                                                model_exp_epoch_index,
                                                model_exp=model_exp,
                                                dataset_exp=dataset_exp,
                                                )
            analyze_result = model_executor.run()
            fig_drawer = self.draw_performance(analyze_result,dynamics)
            fig_file_name = self.get_model_performance_fig_name(analyze_result)
            fig_drawer.save_fig(fig_file_name)
            print("draw " + fig_file_name)
        print("PROCESS COMPLETED!\n\n")