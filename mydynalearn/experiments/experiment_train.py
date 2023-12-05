from mydynalearn.model import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.dataset.dynamic_dataset.getter import get as get_dataset
from mydynalearn.model.getter import get as get_model
from mydynalearn.evaluator import *
from mydynalearn.dataset.getter import get as dataset_loader_getter


class ExperimentTrain():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME

        self.network = get_network(config)
        self.dynamics = get_dynamics(config,self.network)
        self.dataset = get_dataset(self.config)
        self.DataSetLoader = dataset_loader_getter(self.config)
        self.model = get_model(config,self.network,self.dynamics)
        self.TASKS = [
            "generate_data",
            "partition_dataSet",
            "train_model",
            # "performance_evaluation",
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

    def generate_data(self):
        if len(os.listdir(self.config.datapath_to_datasets))==0:
            self.network.create_net()
            self.dataset.run(self.network, self.dynamics)
            self.dataset.save_dataset()
        else:
            self.dataset = self.dataset.load_dataset()


    def generate_DanamicProcessData(self,beta):
        self.dynamics.beta = beta
        self.dataset.run_dynamic_process(self.network, self.dynamics)
    def partition_dataSet(self):
        num_test = self.config.dataset.num_test
        train_set, val_set, test_set = self.dataset.split_dataset(num_test)
        self.train_loader = self.DataSetLoader(train_set)
        self.val_loader = self.DataSetLoader(val_set)
        self.test_loader = self.DataSetLoader(test_set)


    def train_model(self, restore_best=True):
        self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
        )