import copy
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
import torch
from mydynalearn.dynamics.util.Simple_Dynamic_weight import Simple_Dynamic_weight
#  进行一步动力学
class sis(MessagePassing):
    def __init__(self,config):
        super().__init__(aggr='add')
        self.config = config
        self.device = self.config.device
        self.name = config.name
        self.eff_infection = config.eff_infection
        self.initSeedFraction = config.initSeedFraction
        self.maxDimension = config.maxDimension
        self.mu = config.recovery
        self.statesMap = {"S": 0, "I": 1}  # [S,I]
        self.num_state = config.num_state
        self.nodeState_map = torch.eye(self.num_state).to(self.device, dtype = torch.long)

    def runOneStep(self, x0, x1, incMatrix_AdjAct1, network):
        self.network = network
        self.num_nodes = network.num_nodes
        self.betaList = (self.eff_infection*self.mu/network.avg_k).to(self.device)
        self.adjActEdges = self.getAdjActivateSimplex(x0, incMatrix_AdjAct1, network)
        x0, x1, y_ob, y_true = self.spread(x0,x1, self.adjActEdges)
        return x0, x1, y_ob, y_true, self.adjActEdges

    def spread(self, x0,x1, adjActEdges):
        old_x0 = copy.deepcopy(x0)
        old_x1 = copy.deepcopy(x1)
        true_tp = torch.zeros(x0.shape).to(self.device)
        s_index = torch.where(x0[:, self.statesMap["S"]] == 1)[0].to(self.device, dtype = torch.long)
        I_index = torch.where(x0[:, self.statesMap["I"]] == 1)[0].to(self.device, dtype = torch.long)
        inf_S_index = self.dy_action_infection(x0,x1, s_index, adjActEdges, true_tp)
        recover_I_index = self.dy_action_recovery(I_index, true_tp)
        new_x0 , new_x1 = self.update_x0(x0, inf_S_index, recover_I_index)
        return old_x0, old_x1, new_x0, true_tp

    def update_x0(self, x0, inf_S_index, recover_I_index):
        if inf_S_index.shape[0] > 0:
            x0[inf_S_index, :] = self.nodeState_map[1]  # S->I
        if recover_I_index.shape[0] > 0:
            x0[recover_I_index, :] = self.nodeState_map[0]  # I->S
        x1 = self.get_x1_from_x0(x0)
        return x0, x1

    def dy_action_recovery(self, I_index, true_tp):
        recover_prob = self.mu * torch.ones(I_index.shape[0]).to(self.device)
        random_p = torch.rand(I_index.shape[0]).to(self.device)
        recover_I_index = I_index[torch.where(random_p <= recover_prob)[0]]
        true_tp[I_index, self.statesMap["I"]] = 1 - self.mu
        true_tp[I_index, self.statesMap["S"]] = self.mu
        return recover_I_index

    def dy_action_infection(self, x0, x1, s_index, adjActEdges, true_tp):
        # 感染概率
        betaList = self.betaList.view(-1, 1).repeat(1, self.num_nodes)
        # 不被感染的概率矩阵
        NIP_Matrix:'maxDimension,num_nodes' = torch.pow(1 - betaList, adjActEdges)
        NIP = torch.prod(NIP_Matrix,dim=0)
        # 感染的概率 IP (infected probability)
        IP = 1 - NIP
        random_p = torch.rand(self.num_nodes).to(self.device)
        # 找出被感染的S态节点
        inf_S_index = torch.where((random_p <= IP) & (x0[:, self.statesMap["S"]] == 1))[0]
        # 修改实际迁移概率
        true_tp[s_index, self.statesMap["I"]] = IP[s_index]
        true_tp[s_index, self.statesMap["S"]] = 1 - IP[s_index]
        return inf_S_index

    def getAdjActivateSimplex(self, x0, incMatrix_AdjAct1, network):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        adjActEdges = torch.sparse.sum(incMatrix_AdjAct1,dim=1).to_dense()

        return adjActEdges
    def get_x1_from_x0(self,x0):
        x1 = torch.sum(x0[self.network.edges], dim=-2)
        return x1