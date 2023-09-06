import loguru

from mydynalearn.model.nn import *
__gnn__ = {
    "GraphAttentionModel": GraphAttentionLayer,
    "SimplicialAttentionModel": SimplexAttentionLayer,
    "SimplicialDiffAttentionModel": SimplexDiffAttentionLayer,
}


def get(config):
    NAME = config.NAME
    if NAME in __gnn__:
        gnn = __gnn__[NAME]
        return gnn
    else:
        loguru.logger.error("there is no Gnn model named {}",NAME)