import torch
from mydynalearn.config.config_drawer import DrawerConfig
from mydynalearn.drawer.matplot_drawer.fig_ytrure_ypred.getter import get as performance_drawer_getter
import os
class MatplotController():
    def __init__(self,analyze_trained_model,analyze_trained_model_to_realnet):
        self.config = DrawerConfig().default()
        self.analyze_trained_model = analyze_trained_model
        self.analyze_trained_model_to_realnet = analyze_trained_model_to_realnet
        self.TASKS = [
            "trained_model_draw",
            "realnet_draw"
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
    def draw_performance(self,result):
        fig_drawer = performance_drawer_getter(result['dynamics'])
        fig_drawer.scatterT(**result)
        return fig_drawer
    def get_model_performance_fig_name(self,result,train_param,epoch_index):
        merged_string = '_'.join((str(value) for value in train_param))
        fig_dir = self.config.model_performance_figdir
        model_performance_figdir = os.path.join(fig_dir,
                                                result['model_name'])
        self.make_dir(model_performance_figdir)
        fig_file_name = os.path.join(model_performance_figdir,
                                     merged_string + '_epoch{:d}.png'.format(epoch_index))
        return fig_file_name
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
        train_params = self.analyze_trained_model.experiment_manager_train.set_train_params()
        for train_param in train_params:
            epoch_index = self.analyze_trained_model.experiment_manager_train.epochs-1
            model_exp = self.analyze_trained_model.experiment_manager_train.get_loaded_model_exp(train_param, epoch_index)
            result = self.analyze_trained_model.load_test_result(model_exp, epoch_index)
            fig_drawer = self.draw_performance(result)
            fig_file_name = self.get_model_performance_fig_name(result,train_param,epoch_index)
            fig_drawer.save_fig(fig_file_name)
            print("draw " + fig_file_name)
        print("PROCESS COMPLETED!\n\n")

    def realnet_draw(self):
        print("*"*10+" DRAW REALNET PERFORMANCE "+"*"*10)
        train_params = self.analyze_trained_model.experiment_manager_train.set_train_params()
        for train_param in train_params:
            epoch_index = self.analyze_trained_model.experiment_manager_train.epochs-1
            train_param_keys = ["ModelNet", "ModelDynamics", "ModelGnn", "ModelIsWeight"]
            train_param_dict = {k: v for k, v in zip(train_param_keys, train_param)}
            dynamics_params = self.analyze_trained_model_to_realnet.experiment_manager_realnet.get_available_realnet_dynamics(train_param[1])
            realnet_params = self.analyze_trained_model_to_realnet.experiment_manager_realnet.set_realnet_params(dynamics_params=dynamics_params)
            for realnet_param in realnet_params:
                realnet_param_keys = ["RealNet", "RealDynamics"]
                realnet_param_dict = {k: v for k, v in zip(realnet_param_keys, realnet_param)}
                real_exp = self.analyze_trained_model_to_realnet.experiment_manager_realnet.get_loaded_real_exp(realnet_param)
                file_path = self.analyze_trained_model_to_realnet.get_test_result_filepath(real_exp, train_param, epoch_index)
                assert os.path.exists(file_path)
                result = self.analyze_trained_model_to_realnet.load_test_result(real_exp,train_param,epoch_index)
                fig_drawer = self.draw_performance(result)
                fig_file_name = self.get_realnet_performance_fig_name(real_exp,train_param,epoch_index)
                fig_drawer.save_fig(fig_file_name)
                print("draw " + fig_file_name)
        print("PROCESS COMPLETED!\n\n")
