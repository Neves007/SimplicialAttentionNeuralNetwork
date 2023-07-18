import loguru

from mydynalearn.model import graphAttentionModel,simplicialAttentionModel
__Model__ = {
    "graphAttentionModel": graphAttentionModel,
    "simplicialAttentionModel": simplicialAttentionModel
}


def get(config):
    name = config.model.name
    if name in __Model__:
        model = __Model__[name](config)
        return model
    else:
        loguru.logger.error("there is no model named {}",name)