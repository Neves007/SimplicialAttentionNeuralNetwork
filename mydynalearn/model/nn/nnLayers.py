from mydynalearn.model.util.InOut_LinearLayers import inLinearLayers,outLinearLayers
from mydynalearn.model.nn import GraphAttentionLayer,SimplexAttentionLayer
def get_node_in_layers(config):
    in_channels = config.in_channels
    device = config.device
    in_layer = inLinearLayers(in_channels).to(device)
    return in_layer
def get_edge_in_layers(config):
    in_channels = config.in_channels
    device = config.device
    in_layer = inLinearLayers(in_channels).to(device)
    return in_layer

def get_gat_layers(config):
    in_features = config.gnn_channels
    out_features = config.gnn_channels
    concat = config.concat
    heads = config.heads
    device = config.device
    gat_layers = GraphAttentionLayer(in_features, out_features, heads, concat).to(device)
    return gat_layers
def get_sat_layers(config):
    in_features = config.gnn_channels
    out_features = config.gnn_channels
    concat = config.concat
    heads = config.heads
    device = config.device
    gat_layers = SimplexAttentionLayer(in_features, out_features, heads, concat).to(device)
    return gat_layers

def get_out_layers(config):
    out_channels = config.out_channels
    heads = config.heads
    device = config.device
    out_layer = outLinearLayers(out_channels,heads).to(device)
    return out_layer