
class performace_data_handler:
    def __init__(self):
        pass
    def _get_performance_index(self,dynamic_data_handler):
        U_U = dynamic_data_handler.get_transfer_index("U", "U")
        U_A = dynamic_data_handler.get_transfer_index("U", "A")
        A_U = dynamic_data_handler.get_transfer_index("A", "U")
        A_A = dynamic_data_handler.get_transfer_index("A", "A")
        performance_index = (U_U,
                             U_A,
                             A_U,
                             A_A)
        return performance_index

    def _get_performance_data(self,dynamic_data_handler):
        # 混淆矩阵
        U_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U","U")
        U_A_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U","A")
        A_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A","U")
        A_A_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A","A")
        performance_data = (U_U_true_pred,
                            U_A_true_pred,
                            A_U_true_pred,
                            A_A_true_pred)
        return performance_data

