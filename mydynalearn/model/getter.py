import loguru

from mydynalearn.model import GraphAttentionModel,SimplicialAttentionModel
__Model__ = {
    "GAT": GraphAttentionModel,
    "SAT": SimplicialAttentionModel,
    "DiffSAT": SimplicialAttentionModel
}


def get(config,dataset):
    NAME = config.model.NAME
    if NAME in __Model__:
        model = __Model__[NAME](config,dataset)
        return model
    else:
        loguru.logger.error("there is no model named {}",NAME)