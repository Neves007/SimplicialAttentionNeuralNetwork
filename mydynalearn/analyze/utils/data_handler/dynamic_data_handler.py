import torch
import itertools
class DynamicDataHandler():
    def __init__(self,dynamics,test_result_time_list):
        self.STATES_MAP = dynamics.STATES_MAP
        self.dynamics = dynamics
        self.test_result_time_list = test_result_time_list

        self.x_T = torch.cat([data["x"] for data in test_result_time_list],dim=0)
        self.y_ob_T = torch.cat([data["y_ob"] for data in test_result_time_list],dim=0)
        self.y_true_T = torch.cat([data["y_true"] for data in test_result_time_list],dim=0)
        self.y_pred_T = torch.cat([data["y_pred"] for data in test_result_time_list],dim=0)
        self.w_T = torch.cat([data["w"] for data in test_result_time_list],dim=0)

        self.state_transitions = self.get_state_transitions()

    def get_state_transitions(self):
        STATES = self.STATES_MAP.keys()
        # 使用 itertools.product 生成状态转换的所有可能组合
        state_transitions = itertools.product(STATES, repeat=2)
        state_transitions = [state_transition for state_transition in state_transitions]
        return state_transitions

    def get_transition_lables(self):
        '''通过对动力学状态的排列组合得出，转换的的标签

        :return:
        '''
        STATES = self.STATES_MAP.keys()
        # 使用 itertools.product 生成状态转换的所有可能组合
        transitions = itertools.product(STATES, repeat=2)
        transition_lables = [f'{start}_to_{end}' for start, end in transitions]
        return transition_lables

    def get_performance_index(self):
        performance_index = [self._get_transfer_index(*state_transition) for state_transition in self.state_transitions]
        return performance_index

    def get_performance_data(self):
        performance_data = [self._get_ytrue_ypred_pair(*state_transition) for state_transition in self.state_transitions]
        return performance_data



    def _get_transfer_index(self,origin_state,trans_state):
        '''
        输入：origin_state节点的原状态，trans_state迁移状态
        输出：执行该类迁移的节点序号
        '''
        index = torch.where((self.x_T[:, self.STATES_MAP[origin_state]] == 1) & (self.y_ob_T[:, self.STATES_MAP[trans_state]] == 1))[0]
        return index

    def _get_ytrue_ypred_pair(self,origin_state,trans_state):
        '''
        输入：origin_state节点的原状态，trans_state迁移状态
        输出：执行该类迁移的节点，真实迁移概率和预测迁移概率对
        '''
        index = self._get_transfer_index(origin_state,trans_state)
        ytrue_ypred_pair = torch.cat((self.y_true_T[index, self.STATES_MAP[trans_state]].view(-1, 1),
                                      self.y_pred_T[index, self.STATES_MAP[trans_state]].view(-1, 1))
                                     , dim=1)
        return ytrue_ypred_pair