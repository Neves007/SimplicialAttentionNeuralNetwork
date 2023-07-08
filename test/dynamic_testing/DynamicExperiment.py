from mydynalearn.nn import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.dataset.getter import get as get_dataset
from mydynalearn.evaluator import *
from mydynalearn.logger.logger import logger


class DynamicExperiment():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.name = config.name

        self.network = get_network(config.network)
        self.dynamics = get_dynamics(config.dynamics)
        self.dataset = get_dataset(config.train_details)
        self.logger = logger()
    def run(self):
        eff_infectionList,stady_rho_list = self.generate_DanamicProcessData()
        self.dynamic_evaluation(eff_infectionList,stady_rho_list)


    def generate_DanamicProcessData(self):
        len_betaList = 51
        eff_infectionList = torch.linspace(0, 2, len_betaList)
        stady_rho_list = 1. * torch.ones(len_betaList)

        for index, eff_infection in enumerate(eff_infectionList):
            stady_rho = self.get_stady_rho(eff_infection)
            stady_rho_list[index] = stady_rho/self.network.num_nodes
        return eff_infectionList,stady_rho_list
    def get_stady_rho(self,eff_infection):
        self.dynamics.eff_infection[0] = eff_infection
        self.dataset.run_dynamicProcess(self.network, self.dynamics)
        node_timeEvolution = self.dataset.y_ob_T

        stady_nodeState = node_timeEvolution[-100:]
        stady_rho = stady_nodeState.sum(dim=-2)[:,1].mean()
        return stady_rho


    def dynamic_evaluation(self,eff_infectionList,stady_rho_list):
        dynamicEvaluator = DynamicEvaluator(self.config)
        dynamicEvaluator.evaluate(eff_infectionList,stady_rho_list)
        self.logger.evaluate()
