from mydynalearn.model import *
from mydynalearn.networks.getter import get as get_network
from mydynalearn.dynamics.getter import get as get_dynamics
from mydynalearn.dataset.dynamic_dataset.getter import get as get_dataset
from mydynalearn.evaluator import *


class DynamicExperiment():
    def __init__(self,config):
        self.config = config
        self.toyExp = True
        self.NAME = config.NAME

        self.network = get_network(config)
        self.dynamics = get_dynamics(config,self.network)
        self.dataset = get_dataset(config,self.network,self.dynamics)
    def run(self):
        eff_infectionList,stady_rho_list = self.generate_DanamicProcessData()
        self.dynamic_evaluation(eff_infectionList,stady_rho_list)


    def generate_DanamicProcessData(self):
        len_beta_list = 101
        eff_infectionList = torch.linspace(0, 2, len_beta_list)
        stady_rho_list = 1. * torch.ones(len_beta_list)

        for index, eff_infection in enumerate(eff_infectionList):
            stady_rho = self.get_stady_rho(eff_infection)
            stady_rho_list[index] = stady_rho/self.network.NUM_NODES
        return eff_infectionList,stady_rho_list
    def get_stady_rho(self,eff_infection):
        self.dynamics.EFF_AWARE_A1[0] = eff_infection
        self.dataset.run_dynamic_process()
        node_timeEvolution = self.dataset.y_ob_T

        stady_nodeState = node_timeEvolution[-50:]
        stady_rho = stady_nodeState.sum(dim=-2)[:,1].mean()
        return stady_rho


    def dynamic_evaluation(self,eff_infectionList,stady_rho_list):
        dynamicEvaluator = DynamicEvaluator(self.config,self.dynamics)
        dynamicEvaluator.evaluate(eff_infectionList,stady_rho_list)
