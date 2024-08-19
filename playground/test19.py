import numpy as np

# 假设 y_pred 是模型的输出，形状为 (10, 3)，每行表示一个概率分布
y_pred = np.array([[0.2, 0.5, 0.3],
                   [0.1, 0.8, 0.1],
                   [0.4, 0.4, 0.2],
                   [0.3, 0.3, 0.4],
                   [0.7, 0.2, 0.1],
                   [0.5, 0.2, 0.3],
                   [0.2, 0.7, 0.1],
                   [0.6, 0.3, 0.1],
                   [0.3, 0.4, 0.3],
                   [0.2, 0.3, 0.5]])

# state_map 数组表示各状态
state_map = np.array(['UU', 'A1U', 'UA2'])

# 使用 numpy.random.choice 从概率分布中采样
samples = np.array([np.random.choice(state_map, p=y_pred[i]) for i in range(len(y_pred))])

print("采样结果:", samples)
