from mydynalearn.model.nn.nnlayers import *
import torch.nn as nn

class SimplicialAttentionModel(nn.Module):
    def __init__(self, config):
        """Dense version of GAT."""
        super(SimplicialAttentionModel,self).__init__()
        self.config = config
        self.in_layer = get_node_in_layers(self.config.model)
        self.sat_layers = get_gnn_layer(self.config.model)
        self.out_layers = get_out_layers(self.config.model)

    def forward(self,network, dynamics, x0, y_ob, y_true, weight):
        # 数据预处理
        x0 = x0.squeeze()
        y_ob = y_ob.squeeze()
        y_true = y_true.squeeze()
        weight = weight.squeeze()
        # 高阶信息
        x1 =dynamics.get_x1_from_x0(x0, network)
        # 只考虑edge_index
        x0_in = self.in_layer(x0)
        x1_in = self.in_layer(x1)
        if network.MAX_DIMENSION ==2:
            x2 =dynamics.get_x2_from_x0(x0, network)
            x2_in = self.in_layer(x2)
            sat_args = {
                "network":network,
                "x0":x0_in,
                "x1":x1_in,
                "x2":x2_in}
        else:
            sat_args = {
                "network": network,
                "x0": x0_in,
                "x1": x1_in}
        x = self.sat_layers(**sat_args)

        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight
