from mydynalearn.nn import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.dataset.getter import get as get_dataset
from mydynalearn.nn.Models import Model
from easydict import EasyDict as edict
from mydynalearn.evaluator import *
from mydynalearn.logger.logger import logger


class Experiment():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.name = config.name

        self.network = get_network(config.network)
        self.dynamics = get_dynamics(config.dynamics)
        assert self.network.maxDimension == self.dynamics.maxDimension == 1
        self.dataset = get_dataset(config.train_details)
        self.logger = logger(config)
        self.model = Model(config,self.logger)
        self.__tasks__ = [
            "generate_data",
            "partition_dataSet",
            "train_model",
            "performance_evaluation",
        ]
    def run(self):
        tasks = self.__tasks__

        for t in tasks:
            if t in self.__tasks__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.__tasks__}`"
                )

    def generate_data(self):
        if len(os.listdir(self.config.path_to_datasets))==0:
            self.dataset.run(self.network, self.dynamics)
            self.saveDataSet()
        else:
            self.loadDataSet()
        self.logger.generate_data(self.network, self.dynamics)

    def saveDataSet(self):
        info = {
        "network" : self.network,
        "dynamics" : self.dynamics,
        "dataset" : self.dataset,
        }
        file_name = self.config.path_to_datasets+"/dataset.pkl"
        with open(file_name, "wb") as file:
            pickle.dump(info,file)

    def loadDataSet(self):
        file_name = self.config.path_to_datasets + "/dataset.pkl"
        with open(file_name, "rb") as file:
            info = pickle.load(file)
        self.network = info["network"]
        self.dynamics = info["dynamics"]
        self.dataset = info["dataset"]

    def generate_DanamicProcessData(self,beta):
        self.dynamics.beta = beta
        self.dataset.run_dynamicProcess(self.network, self.dynamics)

    def partition_dataSet(self):
        testSet_timestep = self.config.train_details.testSet_timestep
        self.testSet = edict({
            "network": self.network,
            "nodeFeature_T":self.dataset.nodeFeature_T[-testSet_timestep:],
            "y_ob_T": self.dataset.y_ob_T[-testSet_timestep:],
            "y_true_T": self.dataset.y_true_T[-testSet_timestep:],
            "weight": self.dataset.weight[-testSet_timestep:],
        })
        self.trainSet = edict({
            "network": self.network,
            "nodeFeature_T":self.dataset.nodeFeature_T[:-testSet_timestep:2],
            "y_ob_T": self.dataset.y_ob_T[:-testSet_timestep:2],
            "y_true_T": self.dataset.y_true_T[:-testSet_timestep:2],
            "weight": self.dataset.weight[:-testSet_timestep:2],
        })
        self.valSet = edict({
            "network": self.network,
            "nodeFeature_T":self.dataset.nodeFeature_T[1:-testSet_timestep:2],
            "y_ob_T": self.dataset.y_ob_T[1:-testSet_timestep:2],
            "y_true_T": self.dataset.y_true_T[1:-testSet_timestep:2],
            "weight": self.dataset.weight[1:-testSet_timestep:2],
        })

    def train_model(self, restore_best=True):
        self.model.fit(
            self.trainSet,
            val_dataset=self.valSet,
            test_dataset=self.testSet,
        )
    def performance_evaluation(self):
        epochEvaluator = EpochEvaluator(self.config)
        epochEvaluator.evaluate()
        self.logger.evaluate()
