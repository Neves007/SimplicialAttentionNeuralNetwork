import os

# 获取配置
from mydynalearn.config import ExperimentConfig
from mydynalearn.experiments import Experiment


def fix_config(config):
    # T总时间步
    config.dataset.num_samples = num_samples
    config.dataset.num_test = testset_timestep
    config.dataset.epochs = epochs  # 10
    # 检查点
    config.dataset.check_first_epoch = check_first_epoch  # 10
    config.dataset.check_first_epoch_maxtime = check_first_epoch_maxtime
    config.dataset.check_first_epoch_timestep = check_first_epoch_timestep
    config.set_path()


def get_experiment(NAME, network, dynamics, dataset, nn_type, weight):
    config = ExperimentConfig.default(
        NAME=NAME,
        network=network,
        dynamics=dynamics,
        dataset=dataset,
        nn_type=nn_type,
        is_weight=weight,
        seed=0
    )
    fix_config(config)
    exp = Experiment(config)
    return exp


num_samples = 10000
testset_timestep = 10
epochs = 30  # 10
check_first_epoch = False  # 10
check_first_epoch_maxtime = 1000
check_first_epoch_timestep = 100


'''
network = ["ER","SCER"]
dynamics = ["UAU","CompUAU","SCUAU","SCCompUAU"]
dataset = ["graph","simplicial"]
'''
# exp = get_experiment(NAME="dynamicLearning-ER-UAU-GAT",
#                      network="ER",
#                      dynamics="UAU",
#                      dataset="graph",
#                      nn_type="graph",
#                      weight=True)
#
# exp = get_experiment(NAME="dynamicLearning-ER-CompUAU-GAT",
#                      network="ER",
#                      dynamics="CompUAU",
#                      dataset="graph",
#                      nn_type="graph",
#                      weight=True)
exp = get_experiment(NAME="dynamicLearning-SCER-SCCompUAU-SAT",
                     network="SCER",
                     dynamics="SCCompUAU",
                     dataset="simplicial",
                     nn_type="simplicial",
                     weight=False)
exp.run()





