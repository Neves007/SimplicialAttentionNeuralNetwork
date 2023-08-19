import torch
from easydict import EasyDict as edict
from mydynalearn.dataset.dynamic_dataset.dynamic_dataset import DynamicDataset

class DynamicDatasetCompUAU(DynamicDataset):
    def __init__(self, config,network,dynamics) -> None:
        super().__init__(config,network,dynamics)

    def save_dnamic_info(self, t, old_x0, old_x1,old_x2, new_x0, true_tp, weight, **kwargs):
        self.x0_T[t] = old_x0
        self.x1_T[t] = old_x1
        self.x2_T[t] = old_x2
        self.y_ob_T[t] = new_x0
        self.y_true_T[t] = true_tp
        self.weight_T[t] = weight

    def set_dynamic_info(self):
        self.x0_T = torch.ones(self.num_samples, self.network.NUM_NODES, self.dynamics.NUM_STATES).to(self.config.device,dtype = torch.float)
        self.x1_T = torch.ones(self.num_samples, self.network.NUM_EDGES, self.dynamics.NUM_STATES).to(self.config.device,dtype = torch.float)
        self.x2_T = torch.ones(self.num_samples, self.network.NUM_TRIANGLES, self.dynamics.NUM_STATES).to(self.config.device,dtype = torch.float)
        self.y_ob_T = torch.ones(self.num_samples, self.network.NUM_NODES, self.dynamics.NUM_STATES).to(self.config.device,dtype = torch.float)
        self.y_true_T = torch.ones(self.num_samples, self.network.NUM_NODES, self.dynamics.NUM_STATES).to(self.config.device,dtype = torch.float)
        self.weight_T = torch.ones(self.num_samples, self.network.NUM_NODES).to(self.config.device, dtype = torch.float)

    def get_dataset_from_index(self,index):
        dataset = edict({
            "network":self.network,
            "x0_T":self.x0_T[index],
            "x1_T":self.x1_T[index],
            "x2_T":self.x2_T[index],
            "y_ob_T":self.y_ob_T[index],
            "y_true_T":self.y_true_T[index],
            "weight":self.weight_T[index]
        })
        return dataset



