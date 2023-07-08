import copy
from torch_geometric.nn import MessagePassing
import torch
#  进行一步动力学
class sir(MessagePassing):
    def __init__(self,config):
        super().__init__(aggr='add')
        self.config = config
        self.name = config.name
        self.beta = config.infection
        self.initSeedFraction = config.initSeedFraction
        self.mu = config.recovery
        self.statesMap = {"S": 0, "I": 1, "R":2}  # [S,I,R]
        self.num_state = config.num_state
        self.nodeState_map = torch.eye(self.num_state).to(self.config.device,dtype = torch.long)

    def runOneStep(self, x, edge_index):
        x = x.to(self.config.device,dtype = torch.long)
        self.neiborDistri = self.getNeiborDistribution(x, edge_index)
        x, y_ob, y_true = self.update_x(x, self.neiborDistri)
        return x, y_ob, y_true

    def update_x(self, x, neiborDistri):
        x_temp = copy.deepcopy(x)
        true_tp = torch.zeros(x.shape).to(self.config.device)
        s_index = torch.where(x[:, self.statesMap["S"]] == 1)[0].to(self.config.device)
        I_index = torch.where(x[:, self.statesMap["I"]] == 1)[0].to(self.config.device)
        R_index = torch.where(x[:, self.statesMap["R"]] == 1)[0].to(self.config.device)
        inf_S_index = self.dy_action_infection(s_index, neiborDistri, true_tp)
        recover_I_index = self.dy_action_recovery(I_index, true_tp)
        self.update_state(x, inf_S_index, recover_I_index)
        return x_temp, x, true_tp

    def update_state(self, x, inf_S_index, recover_I_index):
        if inf_S_index.shape[0] > 0:
            x[inf_S_index, :] = self.nodeState_map[self.statesMap["I"]]  # S->I
        if recover_I_index.shape[0] > 0:
            x[recover_I_index, :] = self.nodeState_map[self.statesMap["R"]]  # I->R
        return x

    def dy_action_recovery(self, I_index, true_tp):
        recover_prob = self.mu * torch.ones(I_index.shape[0])
        random_p = torch.rand(I_index.shape[0])
        recover_I_index = I_index[torch.where(random_p <= recover_prob)[0]]
        true_tp[I_index, self.statesMap["I"]] = 1 - self.mu
        true_tp[I_index, self.statesMap["S"]] = self.mu
        return recover_I_index

    def dy_action_infection(self, s_index, neiborDistri, true_tp):
        infection_prob = (1 - torch.pow(1 - self.beta, neiborDistri[:, self.statesMap["I"]]))[s_index]
        random_p = torch.rand(s_index.shape[0]).to(self.config.device)
        inf_S_index = s_index[torch.where(random_p <= infection_prob)[0]]
        true_tp[s_index, self.statesMap["I"]] = infection_prob
        true_tp[s_index, self.statesMap["S"]] = 1 - infection_prob
        return inf_S_index

    def getNeiborDistribution(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j