import yaml
import torch

# Load the YAML configuration file
with open('network_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# Example of how you might use this configuration data
# Assuming you have a function that requires a tensor
def process_eff_aware(eff_aware_list):
    return torch.tensor(eff_aware_list)

# Accessing and using the configuration for UAU
uau_config = config_data['UAU']

print(uau_config)