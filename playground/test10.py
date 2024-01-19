'''torch 显存管理
'''
import torch



def get_free_memory():
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory
    print('显存容量：{:.2f} MB', free_memory / 1024 ** 2)


def acc_var():
    a = torch.rand(10000, 10000).to(device)

device = torch.device('cuda')
get_free_memory()
print('add tensor')
a = torch.rand(10000,10000).to(device)
get_free_memory()

get_free_memory()

print('add tensor fun')
acc_var()
get_free_memory()