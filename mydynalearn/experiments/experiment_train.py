from mydynalearn.model import *
from mydynalearn.dataset import DynamicDataset
from mydynalearn.model import Model
from mydynalearn.evaluator import *



class ExperimentTrain():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME
        self.dataset = DynamicDataset(self.config)
        self.model = Model(config)
        self.TASKS = [
            "train_model",
        ]

    def create_dataset(self):
        network, dynamics, train_set, val_set, test_set = self.dataset.run()
        return network, dynamics, train_set, val_set, test_set


    def train_model(self):
        print("model name: ",self.NAME)
        if self.model.need_to_train:
            network, dynamics, train_set, val_set, test_set = self.create_dataset()
            print("begin to train model")
            self.model.run(
                network=network,
                dynamics=dynamics,
                train_set = train_set,
                val_set = val_set,
                test_set = test_set,
            )
            print("The model has been trained completely!")
        else:
            print("The model has already been trained!")
        print()


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