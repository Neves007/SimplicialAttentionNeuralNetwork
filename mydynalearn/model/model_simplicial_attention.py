from mydynalearn.model.nn.nnlayers import *
from .model import Model
from mydynalearn.dataset import simplicialDataSetLoader
import torch
class SimplicialAttentionModel(Model):
    def __init__(self, config,network,dynamics):
        """Dense version of GAT."""
        super().__init__(config,network,dynamics)
        self.in_layer = get_node_in_layers(self.model_config)
        self.sat_layers = get_gnn_layer(self.model_config)

    def forward(self, x0,x1,y_true,y_ob,weight,network,**kwargs):
        # 只考虑edge_index
        dynamics = self.dynamics
        x0_in = self.in_layer(x0)
        x1_in = self.in_layer(x1)
        if network.MAX_DIMENSION ==2:
            x2 = kwargs['x2']
            x2_in = self.in_layer(x2)
            sat_args = {
                "network":network,
                "dynamics":dynamics,
                "x0":x0_in,
                "x1":x1_in,
                "x2":x2_in}
        else:
            sat_args = {
                "network": network,
                "dynamics": dynamics,
                "x0": x0_in,
                "x1": x1_in}
        x = self.sat_layers(**sat_args)

        out = self.out_layers(x)
        return x0,out,y_true,y_ob, weight
