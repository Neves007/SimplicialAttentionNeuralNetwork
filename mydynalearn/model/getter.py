import loguru

from mydynalearn.model import graphAttentionModel,simplicialAttentionModel
__Model__ = {
    "graphAttentionModel": graphAttentionModel,
    "simplicialAttentionModel": simplicialAttentionModel
}


def get(config,network,dynamics):
    NAME = config.model.NAME
    if NAME in __Model__:
        model = __Model__[NAME](config,network,dynamics)
        return model
    else:
        loguru.logger.error("there is no model named {}",NAME)