from mydynalearn.dataset import *
from mydynalearn.model import Model


class ExperimentTrain:
    def __init__(self, config):
        self.config = config
        self.exp_info = self._get_exp_info()
        self.NAME = config.NAME
        self.network = get_network(self.config)
        self.dynamics = get_dynamics(self.config)
        self.dataset = DynamicDataset(self.config)
        self.model = Model(config)
        self.TASKS = [
            "create_dynamic_dataset",
            "train_model",
        ]

    def _get_exp_info(self):
        """
        获取全局信息，包括模型和数据集的相关配置
        """
        return {
            "model_network_name": self.config.network.NAME,
            "model_dynamics_name": self.config.dynamics.NAME,
            "dataset_network_name": self.config.network.NAME,
            "dataset_dynamics_name": self.config.dynamics.NAME,
            "model_name": self.config.model.NAME,
        }

    def create_dynamic_dataset(self):
        """
        创建常规的动力学数据集
        """
        return self.dataset.run()



    def train_model(self):
        """
        训练模型，如果需要训练
        """
        print("Model name:", self.NAME)
        if self.model.need_to_train:
            network, dynamics, train_set, val_set, test_set = self.create_dynamic_dataset()
            print("Beginning model training...")
            self.model.run(
                network=network,
                dynamics=dynamics,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
            )
            print("The model has been trained completely!")
        else:
            print("The model has already been trained!")
        print()

    def run(self):
        """
        运行实验任务
        """
        for task_name in self.TASKS:
            task_method = getattr(self, task_name, None)
            if callable(task_method):
                task_method()
            else:
                raise ValueError(f"{task_name} is an invalid task, possible tasks are {self.TASKS}")

