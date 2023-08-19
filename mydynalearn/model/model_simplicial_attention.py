from mydynalearn.model.nn.nnlayers import *
from .model import Model
from mydynalearn.dataset import simplicial_DataSetLoader
import torch
class simplicialAttentionModel(Model):
    def __init__(self, config,network,dynamics):
        """Dense version of GAT."""
        super().__init__(config,network,dynamics)
        self.in_layers_node_feature = get_node_in_layers(self.model_config)
        self.in_layers_edge_feature = get_edge_in_layers(self.model_config)
        self.in_layers_triangle_feature = get_edge_in_layers(self.model_config)
        self.sat_layers = get_sat_layer(self.model_config)

    def set_dataset_loader(self,test_dataset, train_dataset, val_dataset):
        self.test_loader = simplicial_DataSetLoader(test_dataset)
        self.train_loader = simplicial_DataSetLoader(train_dataset)
        self.val_loader = simplicial_DataSetLoader(val_dataset)
    def prepare_output(self, data):
        network, x0, x1, x2, y_ob, y_true, weight = data
        if self.is_weight==False:
            weight = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)]).to(self.device)
            weight /= weight.sum()
        y_true = y_true
        y_pred = self.forward(x0,x1,x2,network)
        return x0,y_pred,y_true,y_ob, weight

    def forward(self, x0,x1,x2,network):
        # 只考虑edge_index
        # todo:所有类都修改
        x0 = self.in_layers_node_feature(x0)
        # todo：x0 to x1 and x2 可不可以在这儿转换？
        x1 = self.in_layers_edge_feature(x1)
        x2 = self.in_layers_triangle_feature(x2)

        x = self.sat_layers( x0, x1, x2, network,self.dynamics)
        out = self.out_layers(x)
        return out
