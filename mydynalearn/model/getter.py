import loguru

from mydynalearn.model import GraphAttentionModel,SimplicialAttentionModel
__Model__ = {
    "GraphAttentionModel": GraphAttentionModel,
    "SimplicialAttentionModel": SimplicialAttentionModel,
    "SimplicialDiffAttentionModel": SimplicialAttentionModel
}


def get(config,network,dynamics):
    NAME = config.model.NAME
    if NAME in __Model__:
        model = __Model__[NAME](config,network,dynamics)
        return model
    else:
        loguru.logger.error("there is no model named {}",NAME)