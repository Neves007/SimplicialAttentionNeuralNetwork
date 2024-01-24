'''使用 PyTorch 查找两个张量之间的差异元素
'''
import torch

# 假设 t1 和 t2 是已经定义好的两个张量
tensor_a = torch.tensor([1, 2, 2, 3])
tensor_b = torch.tensor([3, 4, 5])

# Find the unique elements in each tensor to remove any duplicates
unique_a = torch.unique(tensor_a)
unique_b = torch.unique(tensor_b)

# Find the intersection of the two tensors
intersection = unique_a[torch.isin(unique_a, unique_b)]

print(intersection)