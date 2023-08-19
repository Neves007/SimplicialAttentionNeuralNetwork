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
            "generate_data",
            "partition_dataSet",
            "train_model",
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

    def generate_data(self):
        if len(os.listdir(self.config.path_to_datasets))==0:
            self.dataset.run()
            self.dataset.save_dataset()
        else:
            self.dataset = self.dataset.load_dataset()


    def generate_DanamicProcessData(self,beta):
        self.dynamics.beta = beta
        self.dataset.run_dynamic_process(self.network, self.dynamics)

    def partition_dataSet(self):
        num_test = self.config.dataset.num_test
        self.train_set, self.val_set, self.test_set = self.dataset.split_dataset(num_test)


    def train_model(self, restore_best=True):
        self.model.fit(
            self.train_set,
            val_dataset=self.val_set,
            test_dataset=self.test_set,
        )
    def performance_evaluation(self):
        epoch_evaluator = evaluatorEpoch(self.config,self.dynamics)
        epoch_evaluator.evaluate()
