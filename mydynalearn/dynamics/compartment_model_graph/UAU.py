import copy
import torch
import random
from mydynalearn.dynamics.compartment_model_graph.compartment_model_graph import CompartmentModelGraph
#  进行一步动力学
class UAU(CompartmentModelGraph):
    def __init__(self, config):
        super().__init__(config)
        self.EFF_AWARE = self.dynamics_config.EFF_AWARE
        self.RECOVERY = self.dynamics_config.RECOVERY
        self.SEED_FREC = self.dynamics_config.SEED_FREC
        self.STATES_MAP = {"U": 0, "A": 1}
        assert len(self.STATES_MAP.keys())==self.NUM_STATES

    def _init_x0(self):
        x0 = torch.zeros(self.NUM_NODES).to(self.DEVICE,torch.long)
        x0 = self.NODE_FEATURE_MAP[x0]

        NUM_SEED_NODES = int(self.NUM_NODES * self.SEED_FREC)
        AWARE_SEED_INDEX = random.sample(range(self.NUM_NODES), NUM_SEED_NODES)
        x0[AWARE_SEED_INDEX] = self.NODE_FEATURE_MAP[self.STATES_MAP["A"]]
        self.x0=x0

    def _get_adj_activate_simplex(self):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        # inc_matrix_adj_act_edge：（节点数，边数）表示节点i与边j，相邻且j是激活边
        inc_matrix_adj_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1, _threshold_scAct=1, target_state='A')
        # adj_act_edges：（节点数）表示节点i相邻激活边数量
        adj_act_edges = torch.sparse.sum(inc_matrix_adj_act_edge,dim=1).to_dense()

        return adj_act_edges
    def _preparing_spreading_data(self):
        adj_act_edges = self._get_adj_activate_simplex()
        old_x0 = copy.deepcopy(self.x0)
        old_x1 = copy.deepcopy(self.x1)
        true_tp = torch.zeros(self.x0.shape).to(self.DEVICE)
        return old_x0, old_x1, true_tp, adj_act_edges

    def _get_nodeid_for_each_state(self):
        U_index = torch.where(self.x0[:, self.STATES_MAP["U"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A_index = torch.where(self.x0[:, self.STATES_MAP["A"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        return U_index, A_index
    def _get_new_feature(self, x0, aware_index, recover_A_index):
        if aware_index.shape[0] > 0:
            x0[aware_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["A"]]
        if recover_A_index.shape[0] > 0:
            x0[recover_A_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["U"]]
        x1 = self.get_x1_from_x0(x0,self.network)
        return x0, x1


    def _dynamic_for_node_A(self, A_index, true_tp):
        recover_prob = self.RECOVERY * torch.ones(self.NUM_NODES).to(self.DEVICE)
        random_p = torch.rand(self.NUM_NODES).to(self.DEVICE)
        recover_A_index = torch.where((random_p <= recover_prob) & (self.x0[:, self.STATES_MAP["A"]] == 1))[0]
        true_tp[A_index, self.STATES_MAP["U"]] = self.RECOVERY
        true_tp[A_index, self.STATES_MAP["A"]] = 1 - self.RECOVERY
        return recover_A_index

    def _dynamic_for_node_U(self, U_index, adj_act_edges, true_tp):
        # 感染概率
        BETA_LIST = self.BETA_LIST.view(-1, 1).repeat(1, self.NUM_NODES)
        # 不被感染的概率矩阵
        not_aware_prob_matrix:'MAX_DIMENSION,NUM_NODES' = torch.pow(1 - BETA_LIST, adj_act_edges)
        not_aware_prob = torch.prod(not_aware_prob_matrix,dim=0)
        # aware_prob (Aware probability)
        aware_prob = 1 - not_aware_prob
        random_p = torch.rand(self.NUM_NODES).to(self.DEVICE)
        # 找出被感染的S态节点
        aware_index = torch.where((random_p <= aware_prob) & (self.x0[:, self.STATES_MAP["U"]] == 1))[0]
        # 修改实际迁移概率
        true_tp[U_index, self.STATES_MAP["U"]] = 1 - aware_prob[U_index]
        true_tp[U_index, self.STATES_MAP["A"]] = aware_prob[U_index]
        return aware_index

    def _spread(self):
        old_x0, old_x1, true_tp, adj_act_edges = self._preparing_spreading_data()
        U_index, A_index = self._get_nodeid_for_each_state()
        aware_index = self._dynamic_for_node_U(U_index, adj_act_edges, true_tp)
        recover_A_index = self._dynamic_for_node_A(A_index, true_tp)
        new_x0 , new_x1 = self._get_new_feature(self.x0, aware_index, recover_A_index)
        weight_args = {
            "DEVICE":self.DEVICE,
            "old_x0":old_x0,
            "adj_act_edges":adj_act_edges,
            "new_x0":new_x0,
            "network":self.network,
            "dynamics":self
        }
        weight = self.get_weight(**weight_args)
        spread_result = {
            "old_x0":old_x0,
            "old_x1":old_x1,
            "new_x0":new_x0,
            "new_x1":new_x1,
            "true_tp":true_tp,
            "weight":weight
        }
        self.set_spread_result(spread_result)
    def _run_onestep(self):
        self.BETA_LIST = (self.EFF_AWARE * self.RECOVERY / self.network.AVG_K).to(self.DEVICE)
        self._spread()

