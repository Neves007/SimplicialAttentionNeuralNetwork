'''layer nomalization
'''
import torch
input = torch.arange(0,12).to(torch.float).reshape(3,4)
output = torch.layer_norm(input, [3,])
print(input)
print(output)