from mydynalearn.model.util.inout_linear_layers import inLinearLayers,outLinearLayers
from mydynalearn.model.nn import GraphAttentionLayer,SimplexAttentionLayer
from mydynalearn.model.nn.getter import get as gnn_getter
def get_node_in_layers(config):
    in_channels = config.in_channels
    DEVICE = config.DEVICE
    in_layer = inLinearLayers(in_channels).to(DEVICE)
    return in_layer

def get_gnn_layer(config):
    in_features = config.gnn_channels
    out_features = config.gnn_channels
    concat = config.concat
    heads = config.heads
    DEVICE = config.DEVICE
    GNNLayers = gnn_getter(config)
    gnn_layers = GNNLayers(in_features, out_features, heads, concat).to(DEVICE)
    return gnn_layers

def get_out_layers(config):
    out_channels = config.out_channels
    heads = config.heads
    DEVICE = config.DEVICE
    out_layer = outLinearLayers(out_channels, heads).to(DEVICE)
    return out_layer