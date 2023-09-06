from mydynalearn.model import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.dataset.dynamic_dataset.getter import get as get_dataset
from mydynalearn.model.getter import get as get_model
from mydynalearn.evaluator import *


class Experiment():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME

        self.network = get_network(config)
        self.dynamics = get_dynamics(config,self.network)
        self.dataset = get_dataset(config,self.network,self.dynamics)
        self.model = get_model(config,self.network,self.dynamics)
        self.__TASKS__ = [
            "performance_evaluation",
        ]
    def run(self):
        tasks = self.__TASKS__

        for t in tasks:
            if t in self.__TASKS__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.__TASKS__}`"
                )

    def performance_evaluation(self):
        epoch_evaluator = evaluatorEpoch(self.config,self.dynamics)
        epoch_evaluator.evaluate()

    def maxR_evaluation(self):
        epoch_evaluator = evaluatorEpoch(self.config,self.dynamics)
        epoch_evaluator.draw_maxR()