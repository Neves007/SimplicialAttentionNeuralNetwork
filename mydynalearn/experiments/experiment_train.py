from mydynalearn.model import *
from mydynalearn.dataset import DynamicDataset
from mydynalearn.model.getter import get as get_model
from mydynalearn.evaluator import *
from torch.utils.data import DataLoader


class ExperimentTrain():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME
        self.dataset = DynamicDataset(self.config)
        self.TASKS = [
            "generate_data",
            "partition_dataSet",
            "set_model",
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
            self.dataset.run()
            self.dataset.save_dataset()
        else:
            self.dataset = self.dataset.load_dataset()


    def generate_DanamicProcessData(self,beta):
        self.dynamics.beta = beta
        self.dataset.run_dynamic_process(self.network, self.dynamics)


    # 放进数据集类里面
    def partition_dataSet(self):
        test_size = self.config.dataset.NUM_TEST
        val_size = int((len(self.dataset)-test_size)/2)
        train_size = len(self.dataset)-test_size-val_size
        train_set, val_set, test_set = torch.utils.data.random_split(self.dataset, [train_size, val_size,test_size])

        self.train_loader = DataLoader(train_set,shuffle=True)
        self.val_loader = DataLoader(val_set,shuffle=True)
        self.test_loader = DataLoader(test_set,shuffle=True)
    def set_model(self):
        self.model = get_model(self.config,self.dataset)

    def train_model(self, restore_best=True):
        self.model.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
        )