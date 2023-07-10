import loguru

from mydynalearn.model import graphAttentionModel
__Model__ = {
    "graphAttentionModel": graphAttentionModel,
}


def get(config):
    name = config.model.name
    if name in __Model__:
        model = __Model__[name](config)
        return model
    else:
        loguru.logger.error("there is no model named {}",name)