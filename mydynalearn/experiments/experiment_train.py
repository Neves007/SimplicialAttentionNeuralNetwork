from mydynalearn.model import *
from mydynalearn.dataset import DynamicDataset
from mydynalearn.model import Model
from mydynalearn.evaluator import *



class ExperimentTrain():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME
        self.model = Model(config)
        self.TASKS = [
            "create_dataset",
            "train_model",
        ]

    def create_dataset(self):
        self.dataset = DynamicDataset(self.config)
        self.dataset.run()

    def train_model(self):
        if self.model.need_to_train:
            print("begin to train model")
            self.model.run(
                dataset = self.dataset,
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