import os

from mydynalearn.model import *
from mydynalearn.dataset import TestDynamicDataset
from mydynalearn.drawer.matplot_drawer.fig_beta_rho.fig_beta_rho import FigBetaRho
from mydynalearn.drawer.matplot_drawing_task.matplot_drawing_task import FigBetaRhoDrawingTask


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
        fig_beta_rho_drawing_task = FigBetaRhoDrawingTask(self.dataset)
        fig_beta_rho_drawing_task.run()


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