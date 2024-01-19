from mydynalearn.model import *
from mydynalearn.dataset import TestDynamicDataset
from mydynalearn.model import Model
from mydynalearn.evaluator import *



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
        self.dataset.run()

    def draw(self):
        stady_rho_list = self.dataset.stady_rho_list
        EFF_BETA_LIST = self.dataset.EFF_BETA_LIST
        dynamicEvaluator = DynamicEvaluator(self.config,self.dynamics)
        dynamicEvaluator.evaluate(EFF_BETA_LIST, stady_rho_list)


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