import numpy as np

# 定义张量 a 和 b
a = np.random.randn(80).reshape((40,2))
b = a.reshape((2,20,2))

R1 = np.corrcoef(a[:,0],a[:,1])[0,1]


R2_0 = np.corrcoef(b[0,:,0],b[0,:,1])[0,1]
R2_1 = np.corrcoef(b[1,:,0],b[1,:,1])[0,1]

print("R1:",R1)
print("R2_0:",R2_0)
print("R2_1:",R2_1)
print("np.sqrt(R2_0 * R2_1):",2 * R2_0 * R2_1 / (R2_0 + R2_1))


