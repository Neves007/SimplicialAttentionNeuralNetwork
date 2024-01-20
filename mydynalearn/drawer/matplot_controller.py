import torch
from mydynalearn.analyze.analyzer import *
from mydynalearn.config import *
from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred.fig_ytrure_ypred import FigYtrureYpred
import os
class MatplotController():
    def __init__(self,analyze_manager):
        config_drawer = Config.get_config_drawer()
        self.config = config_drawer['default']
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
    def draw_performance(self,analyze_result,STATES_MAP):
        fig_drawer = FigYtrureYpred(STATES_MAP)
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
        analyze_result_generator = self.analyze_manager.get_analyze_result_generator_for_best_epoch()
        for analyze_result in analyze_result_generator:
            STATES_MAP = analyze_result['model_dynamics_state_map']
            fig_drawer = self.draw_performance(analyze_result,STATES_MAP)
            fig_file_name = self.get_model_performance_fig_name(analyze_result)
            fig_drawer.save_fig(fig_file_name)
            print("draw " + fig_file_name)
        print("PROCESS COMPLETED!\n\n")