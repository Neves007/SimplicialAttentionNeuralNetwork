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

    def runOneStep(self, nodesState, source_simplicesActivation_Dict, network):
        # 邻居单纯性激活状态情况
        self.neighborSimplexActivation_Matrix: 'maxDimension, num_nodes, Activation' = None
        self.num_nodes = network.num_nodes
        self.betaList = (self.eff_infection*self.mu/network.real_K).to(self.device)
        self.neighborSimplexActivation_Matrix = self.getNeiborSimplex(nodesState, source_simplicesActivation_Dict, network)
        x, y_ob, y_true = self.update_nodesState(nodesState, self.neighborSimplexActivation_Matrix)
        return x, y_ob, y_true,self.neighborSimplexActivation_Matrix

    def update_nodesState(self, nodesState, neighborSimplexActivation_Matrix):
        nodesState_temp = copy.deepcopy(nodesState)
        true_tp = torch.zeros(nodesState.shape).to(self.device)
        s_index = torch.where(nodesState[:, self.statesMap["S"]] == 1)[0].to(self.device,dtype = torch.long)
        I_index = torch.where(nodesState[:, self.statesMap["I"]] == 1)[0].to(self.device,dtype = torch.long)
        inf_S_index = self.dy_action_infection(nodesState,s_index, neighborSimplexActivation_Matrix, true_tp)
        recover_I_index = self.dy_action_recovery(I_index, true_tp)
        self.update_state(nodesState, inf_S_index, recover_I_index)
        return nodesState_temp, nodesState, true_tp

    def update_state(self, nodesState, inf_S_index, recover_I_index):
        if inf_S_index.shape[0] > 0:
            nodesState[inf_S_index, :] = self.nodeState_map[1]  # S->I
        if recover_I_index.shape[0] > 0:
            nodesState[recover_I_index, :] = self.nodeState_map[0]  # I->S

    def dy_action_recovery(self, I_index, true_tp):
        recover_prob = self.mu * torch.ones(I_index.shape[0]).to(self.device)
        random_p = torch.rand(I_index.shape[0]).to(self.device)
        recover_I_index = I_index[torch.where(random_p <= recover_prob)[0]]
        true_tp[I_index, self.statesMap["I"]] = 1 - self.mu
        true_tp[I_index, self.statesMap["S"]] = self.mu
        return recover_I_index

    def dy_action_infection(self,nodesState, s_index, neighborSimplexActivation_Matrix, true_tp):
        # 感染概率
        betaList = self.betaList.view(-1, 1).repeat(1, self.num_nodes)
        # 激活态单纯形数量
        neighborActivateSimplex_Matrix = neighborSimplexActivation_Matrix[:, :, -1]
        # 不被感染的概率矩阵
        NIP_Matrix:'maxDimension,num_nodes' = torch.pow(1 - betaList, neighborActivateSimplex_Matrix)
        NIP = torch.prod(NIP_Matrix,dim=0)
        # 感染的概率 IP (infected probability)
        IP = 1 - NIP
        random_p = torch.rand(self.num_nodes).to(self.device)
        # 找出被感染的S态节点
        inf_S_index = torch.where((random_p <= IP) & (nodesState[:, self.statesMap["S"]] == 1))[0]
        # 修改实际迁移概率
        true_tp[s_index, self.statesMap["I"]] = IP[s_index]
        true_tp[s_index, self.statesMap["S"]] = 1 - IP[s_index]
        return inf_S_index

    def getNeiborSimplex(self, nodesState, source_simplicesActivation_Dict, network):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        neighborSimplexActivation_Matrix = torch.zeros((self.maxDimension,self.num_nodes,2),device=self.device,dtype=torch.long)

        for index_dimension in range(self.maxDimension):
            dimension = index_dimension + 1
            key = "{}-simplex".format(dimension)
            source_simplicesActivation = source_simplicesActivation_Dict[key]
            target_nodes = network.simplices_incidence[key].target_nodes
            temporarySource = torch.arange(target_nodes.shape[0]).to(self.device)
            # 从信息源simplex，传递到目标节点
            propagate_index = torch.stack((temporarySource,target_nodes))
            # aggregateSimplexInfo 存储聚合的单纯形，表示周围有多少个激活态单纯形
            propagate_info, propagate_nodes = self.propagate(propagate_index, source_simplicesActivation)
            neighborSimplexActivation_Matrix[index_dimension] = propagate_info
        return neighborSimplexActivation_Matrix

    def propagate(self, propagate_index,simplicesActivation):
        out = self.message(simplicesActivation, propagate_index)
        out,target_nodes = self.aggregate(out, propagate_index)
        out = self.update(out)
        return out,target_nodes

    def message(self, simplicesActivation, propagate_index):
        source_simplices, target_nodes = propagate_index
        return simplicesActivation[source_simplices]

    def aggregate(self, x_j, propagate_index):
        source_simplices, target_nodes = propagate_index
        out = torch.zeros(self.num_nodes,x_j.shape[-1]).to(self.device,torch.float)
        aggr_out = scatter(x_j, target_nodes, dim=-2, reduce='sum')
        target_nodes = torch.arange(target_nodes.min(), target_nodes.max()+1).to(self.device,torch.long)
        out[target_nodes] = aggr_out
        # 补零
        return out,target_nodes

    def update(self, aggr_out):
        return aggr_out
