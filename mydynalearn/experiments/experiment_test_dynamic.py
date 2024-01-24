import os

from mydynalearn.model import *
from mydynalearn.dataset import TestDynamicDataset
from mydynalearn.drawer.matplot_drawer.fig_beta_rho.fig_beta_rho import FigBetaRho


class ExperimentTestDynamic():
    def __init__(self,config):
        self.config = config
        self.NAME = config.NAME
        self.dataset = TestDynamicDataset(self.config)
        self.TASKS = [
            "run_dynamic_process",
            "draw"
        ]

    def run_dynamic_process(self):
        self.dataset.show_info()
        if self.dataset.is_need_to_run:
            self.dataset.run()
        else:
            self.dataset = self.dataset.load_dataset()

    def draw(self):
        data = self.dataset.get_draw_data()
        drawer = FigBetaRho(self.dataset.dynamics,**data)
        drawer.draw()
        fig_dir_path = self.config.fig_dir_path
        dataset_info = self.dataset.get_info()
        fig_name = "_".join([dataset_info['network_name'],dataset_info['dynamic_name'],'.jpg'])
        fig_file_path = os.path.join(fig_dir_path,fig_name)
        drawer.save_fig(fig_file_path)



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