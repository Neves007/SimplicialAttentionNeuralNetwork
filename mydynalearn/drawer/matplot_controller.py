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

    def get_model_performance_fig_name(self,analyze_result):
        model_info = str.join('_',[analyze_result["model_network_name"],
                                   analyze_result["model_dynamics_name"],
                                   analyze_result["model_name"]
                                   ])
        testdata_info = str.join('_',[analyze_result["dataset_network_name"],
                                      analyze_result["dataset_dynamics_name"],
                                      analyze_result["model_name"]
                                      ])
        fig_name = "MODEL_{}_DATA_{}_epoch_{}".format(model_info,testdata_info,analyze_result['model_exp_epoch_index'])
        fig_dir = self.config.model_performance_figdir
        model_performance_figdir = os.path.join(fig_dir, analyze_result['model_name'])
        self.make_dir(model_performance_figdir)
        fig_file_path = os.path.join(model_performance_figdir,fig_name)
        return fig_file_path

    def trained_model_draw(self):
        print("*"*10+"DRAW TRAIN PERFORMANCE"+"*"*10)
        analyze_result_generator = self.analyze_manager.get_analyze_result_generator_for_best_epoch()
        for analyze_result in analyze_result_generator:
            fig_drawer = FigYtrureYpred(**analyze_result)
            fig_drawer.scatterT()
            fig_file_name = self.get_model_performance_fig_name(analyze_result)
            fig_drawer.save_fig(fig_file_name)
            print("draw " + fig_file_name)
        print("PROCESS COMPLETED!\n\n")