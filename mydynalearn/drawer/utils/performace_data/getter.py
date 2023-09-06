def get_performance_index_UAU(dynamic_data_handler):
    U_U = dynamic_data_handler.get_transfer_index("U", "U")
    U_A = dynamic_data_handler.get_transfer_index("U", "A")
    A_U = dynamic_data_handler.get_transfer_index("A", "U")
    A_A = dynamic_data_handler.get_transfer_index("A", "A")
    performance_index = [U_U,
                         U_A,
                         A_U,
                         A_A]
    return performance_index



def get_performance_index_compUAU(dynamic_data_handler):
    U_U = dynamic_data_handler.get_transfer_index("U", "U")
    U_A1 = dynamic_data_handler.get_transfer_index("U", "A1")
    U_A2 = dynamic_data_handler.get_transfer_index("U", "A2")
    A1_A1 = dynamic_data_handler.get_transfer_index("A1", "A1")
    A1_U = dynamic_data_handler.get_transfer_index("A1", "U")
    A2_A2 = dynamic_data_handler.get_transfer_index("A2", "A2")
    A2_U = dynamic_data_handler.get_transfer_index("A2", "U")
    performance_index = [U_U,
                         U_A1,
                         U_A2,
                         A1_A1,
                         A1_U,
                         A2_A2,
                         A2_U]
    return performance_index

def get_performance_data_UAU(dynamic_data_handler):
    # 混淆矩阵
    U_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U", "U")
    U_A_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U", "A")
    A_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A", "U")
    A_A_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A", "A")
    performance_data = [U_U_true_pred,
                        U_A_true_pred,
                        A_U_true_pred,
                        A_A_true_pred]
    return performance_data
def get_performance_data_compUAU(dynamic_data_handler):
    # 混淆矩阵
    U_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U", "U")
    U_A1_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U", "A1")
    U_A2_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("U", "A2")
    A1_A1_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A1", "A1")
    A1_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A1", "U")
    A2_A2_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A2", "A2")
    A2_U_true_pred = dynamic_data_handler.get_ytrue_ypred_pair("A2", "U")
    performance_data = [U_U_true_pred,
                        U_A1_true_pred,
                        U_A2_true_pred,
                        A1_A1_true_pred,
                        A1_U_true_pred,
                        A2_A2_true_pred,
                        A2_U_true_pred]
    return performance_data



__dynamics__ = {
    "UAU": (get_performance_index_UAU,get_performance_data_UAU),
    "CompUAU": (get_performance_index_compUAU,get_performance_data_compUAU),
    "SCUAU": (get_performance_index_UAU,get_performance_data_UAU),
    "SCCompUAU": (get_performance_index_compUAU,get_performance_data_compUAU),
}


def get(config):
    NAME = config.dynamics.NAME
    if NAME in __dynamics__:
        return __dynamics__[NAME]
    else:
        raise ValueError(
            f"{NAME} is invalid, possible entries are {list(__dynamics__.keys())}"
        )


