import copy
import torch
import random
from mydynalearn.dynamics.compartment_model import CompartmentModel
#  进行一步动力学
class ToySCCompUAU(CompartmentModel):
    def __init__(self,config):
        super().__init__(config)
        # todo: 修改 torch.tensor
        self.EFF_AWARE_A1 = torch.tensor(self.dynamics_config.EFF_AWARE_A1)
        self.EFF_AWARE_A2 = torch.tensor(self.dynamics_config.EFF_AWARE_A2)
        self.RECOVERY = self.dynamics_config.RECOVERY
        self.SEED_FREC_A1 = self.dynamics_config.SEED_FREC_A1
        self.SEED_FREC_A2 = self.dynamics_config.SEED_FREC_A2
        self.STATES_MAP = {"U": 0, "A1": 1, "A2": 2}
        assert len(self.STATES_MAP.keys())==self.NUM_STATES

    def _init_x0(self):
        x0 = torch.zeros(self.NUM_NODES).to(self.DEVICE,torch.long)
        x0 = self.NODE_FEATURE_MAP[x0]

        x0[[0]] = self.NODE_FEATURE_MAP[self.STATES_MAP["U"]]
        x0[[2,6,7]] = self.NODE_FEATURE_MAP[self.STATES_MAP["A2"]]
        x0[[1,3,4,5]] = self.NODE_FEATURE_MAP[self.STATES_MAP["A1"]]
        self.x0=x0

    def _get_adj_activate_simplex(self):
        '''
        聚合邻居单纯形信息
        了解节点周围单纯形，【非激活态，激活态】数量
        '''
        # inc_matrix_adj_act_edge：（节点数，边数）表示节点i与边j，相邻且j是激活边
        inc_matrix_adj_A1_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                      _threshold_scAct=1,
                                                      target_state='A1',
                                                      inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_A2_act_edge = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x1,
                                                      _threshold_scAct=1,
                                                      target_state='A2',
                                                      inc_matrix_adj=self.network.inc_matrix_adj1)
        inc_matrix_adj_A1_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                      _threshold_scAct=2,
                                                      target_state='A1',
                                                      inc_matrix_adj=self.network.inc_matrix_adj2)
        inc_matrix_adj_A2_act_triangle = self.get_inc_matrix_adjacency_activation(inc_matrix_col_feature=self.x2,
                                                      _threshold_scAct=2,
                                                      target_state='A2',
                                                      inc_matrix_adj=self.network.inc_matrix_adj2)
        # adj_act_edges：（节点数）表示节点i相邻激活边数量
        adj_A1_act_edge = torch.sparse.sum(inc_matrix_adj_A1_act_edge,dim=1).to_dense()
        adj_A2_act_edge = torch.sparse.sum(inc_matrix_adj_A2_act_edge,dim=1).to_dense()
        adj_A1_act_triangle = torch.sparse.sum(inc_matrix_adj_A1_act_triangle,dim=1).to_dense()
        adj_A2_act_triangle = torch.sparse.sum(inc_matrix_adj_A2_act_triangle,dim=1).to_dense()

        return adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle

    def _preparing_spreading_data(self):
        adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle = self._get_adj_activate_simplex()
        old_x0 = copy.deepcopy(self.x0)
        old_x1 = copy.deepcopy(self.x1)
        old_x2 = copy.deepcopy(self.x2)
        true_tp = torch.zeros(self.x0.shape).to(self.DEVICE)
        return old_x0, old_x1, old_x2, true_tp, adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle

    def _get_nodeid_for_each_state(self):
        U_index = torch.where(self.x0[:, self.STATES_MAP["U"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A1_index = torch.where(self.x0[:, self.STATES_MAP["A1"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        A2_index = torch.where(self.x0[:, self.STATES_MAP["A2"]] == 1)[0].to(self.DEVICE, dtype=torch.long)
        return U_index, A1_index, A2_index

    def _get_new_feature(self, x0, aware_A1_index, aware_A2_index, recover_A1_index, recover_A2_index):
        if aware_A1_index.shape[0] > 0:
            x0[aware_A1_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["A1"]]  # U->A1
        if aware_A2_index.shape[0] > 0:
            x0[aware_A2_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["A2"]]  # U->A2
        if recover_A1_index.shape[0] > 0:
            x0[recover_A1_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["U"]]  # A1->U
        if recover_A2_index.shape[0] > 0:
            x0[recover_A2_index, :] = self.NODE_FEATURE_MAP[self.STATES_MAP["U"]]  # A2->U
        x1 = self.get_x1_from_x0(x0,self.network)
        x2 = self.get_x2_from_x0(x0,self.network)
        return x0, x1, x2

    def _dynamic_for_node_A1(self, A1_index, true_tp):
        recover_prob = self.RECOVERY * torch.ones(A1_index.shape[0]).to(self.DEVICE)
        random_p = torch.rand(A1_index.shape[0]).to(self.DEVICE)
        recover_A1_index = A1_index[torch.where(random_p <= recover_prob)[0]]
        true_tp[A1_index, self.STATES_MAP["U"]] = self.RECOVERY
        true_tp[A1_index, self.STATES_MAP["A1"]] = 1 - self.RECOVERY
        return recover_A1_index
    def _dynamic_for_node_A2(self, A2_index, true_tp):
        recover_prob = self.RECOVERY * torch.ones(A2_index.shape[0]).to(self.DEVICE)
        random_p = torch.rand(A2_index.shape[0]).to(self.DEVICE)
        recover_A2_index = A2_index[torch.where(random_p <= recover_prob)[0]]
        true_tp[A2_index, self.STATES_MAP["U"]] = self.RECOVERY
        true_tp[A2_index, self.STATES_MAP["A2"]] = 1 - self.RECOVERY
        return recover_A2_index

    def _dynamic_for_node_U(self,U_index, adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle, true_tp):
        # 感染概率
        BETA_LIST_A1 = self.BETA_LIST_A1.view(-1, 1).repeat(1, self.NUM_NODES)
        BETA_LIST_A2 = self.BETA_LIST_A2.view(-1, 1).repeat(1, self.NUM_NODES)
        # 不被感染的概率矩阵
        adjacency_activation_simplex_A1 = torch.stack((adj_A1_act_edge, adj_A1_act_triangle), dim=0)
        adjacency_activation_simplex_A2 = torch.stack((adj_A2_act_edge, adj_A2_act_triangle), dim=0)
        not_aware_A1_prob_matrix:'MAX_DIMENSION,NUM_NODES' = torch.pow(1 - BETA_LIST_A1, adjacency_activation_simplex_A1)
        not_aware_A2_prob_matrix:'MAX_DIMENSION,NUM_NODES' = torch.pow(1 - BETA_LIST_A2, adjacency_activation_simplex_A2)
        # U_A1：与所有A1态邻居接触，并不被影响的概率
        # U_A2：与所有A2态邻居接触，并不被影响的概率
        unaware_A1_prob = torch.prod(not_aware_A1_prob_matrix,dim=0)
        unaware_A2_prob = torch.prod(not_aware_A2_prob_matrix,dim=0)
        unaware_prob = unaware_A1_prob * unaware_A2_prob
        # g_A1：与所有A1态邻居接触，但被影响成为A1态的概率
        # g_A2：与所有A2态邻居接触，但被影响成为A2态的概率
        g_A1 = 1 - unaware_A1_prob + 1e-15
        g_A2 = 1 - unaware_A2_prob + 1e-15
        # f_A1：当已知节点被影响，被影响成为A1态的概率
        # f_A2：当已知节点被影响，被影响成为A2态的概率
        f_A1 = g_A1 * (1 - 0.5 * g_A2) / (g_A1 * (1 - 0.5 * g_A2) + g_A2 * (1 - 0.5 * g_A1))
        f_A2 = g_A2 * (1 - 0.5 * g_A1) / (g_A1 * (1 - 0.5 * g_A2) + g_A2 * (1 - 0.5 * g_A1))
        # 影响过程
        random_p = torch.rand(self.NUM_NODES).to(self.DEVICE)
        aware_prob = 1 - unaware_prob  # 至少被一个A1或A2的邻居影响
        aware_index = torch.where((random_p <= aware_prob) & (self.x0[:, self.STATES_MAP["U"]] == 1))[0]

        random_p = torch.rand(self.NUM_NODES).to(self.DEVICE)
        aware_index_bool = torch.zeros(self.NUM_NODES).to(self.DEVICE)
        aware_index_bool[aware_index] = 1
        aware_A1_index = torch.where((random_p <= f_A1) & (aware_index_bool == 1))[0]
        aware_A2_index = torch.where((random_p > f_A1) & (aware_index_bool == 1))[0]
        # 修改实际迁移概率
        true_tp[U_index, self.STATES_MAP["U"]] = unaware_prob[U_index]
        true_tp[U_index, self.STATES_MAP["A1"]] = (aware_prob * f_A1)[U_index]
        true_tp[U_index, self.STATES_MAP["A2"]] = (aware_prob * f_A2)[U_index]
        return aware_A1_index, aware_A2_index

    def get_weight(self,old_x0,
                   adj_A1_act_edge,
                   adj_A2_act_edge,
                   adj_A1_act_triangle,
                   adj_A2_act_triangle,
                   new_x0):

        simple_dynamic_weight = self.SimpleDynamicWeight(self.DEVICE,
                                                         old_x0,
                                                         new_x0,
                                                         adj_A1_act_edge,
                                                         adj_A2_act_edge,
                                                         adj_A1_act_triangle,
                                                         adj_A2_act_triangle,
                                                         self.network,
                                                         self)
        weight = simple_dynamic_weight.get_weight()
        return weight
    def _spread(self,network):
        self.set_network(network)
        old_x0, old_x1, old_x2, true_tp, adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle = self._preparing_spreading_data()
        U_index, A1_index, A2_index = self._get_nodeid_for_each_state()
        aware_A1_index, aware_A2_index = self._dynamic_for_node_U(U_index, adj_A1_act_edge, adj_A2_act_edge, adj_A1_act_triangle, adj_A2_act_triangle, true_tp)
        recover_A1_index = self._dynamic_for_node_A1(A1_index, true_tp)
        recover_A2_index = self._dynamic_for_node_A2(A2_index, true_tp)
        new_x0, new_x1,new_x2 = self._get_new_feature(self.x0,
                                                      aware_A1_index,
                                                      aware_A2_index,
                                                      recover_A1_index,
                                                      recover_A2_index)
        weight = self.get_weight(old_x0,
                                 adj_A1_act_edge,
                                 adj_A2_act_edge,
                                 adj_A1_act_triangle,
                                 adj_A2_act_triangle,
                                 new_x0)

        nan_indices = torch.nonzero(torch.isnan(true_tp))
        if len(nan_indices) > 0:
            print(nan_indices)
            raise
        spread_result = {
            "old_x0":old_x0,
            "old_x1":old_x1,
            "old_x2":old_x2,
            "new_x0":new_x0,
            "new_x1":new_x1,
            "new_x2":new_x2,
            "true_tp":true_tp,
            "weight": weight
        }
        return spread_result

    def _run_onestep(self):
        assert len(self.EFF_AWARE_A1)==len(self.EFF_AWARE_A2)==len(self.network.AVG_K)
        self.BETA_LIST_A1 = (self.EFF_AWARE_A1 * self.RECOVERY / self.network.AVG_K).to(self.DEVICE)
        self.BETA_LIST_A2 = (self.EFF_AWARE_A2 * self.RECOVERY / self.network.AVG_K).to(self.DEVICE)
        self.BETA_LIST_A1 = torch.asarray([0.1,0.2]).to(self.DEVICE)
        self.BETA_LIST_A2 = torch.asarray([0.1,0.2]).to(self.DEVICE)
        spread_result = self._spread()
        return spread_result