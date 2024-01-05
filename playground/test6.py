import numpy as np

r_values = np.array([0.5,0.67,0.9,0.92])
window_size = 2
threshold = 0.01


# 计算每个窗口的标准差
std_values = []
for i in range(len(r_values) - window_size + 1):
    window = r_values[i : i+window_size]
    std = np.std(window)
    std_values.append(std)
std_values = np.array(std_values)
# 找到首次小于 threshold 的索引
index = np.where(std_values < threshold)[0]


print("r_values: ",r_values)
print("std_values: ",std_values)
print("index: ",index)