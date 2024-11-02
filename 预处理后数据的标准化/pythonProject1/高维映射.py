import numpy as np

# 定义高斯径向基核函数
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# 定义数据
a1 = np.array([1, 2])
a2 = np.array([3, 4])

# 定义一个函数，将数据映射到6维空间
def map_to_6d(a1, a2, gamma):
    # 计算核矩阵
    K = np.zeros((2, 2))
    K[0, 0] = rbf_kernel(a1, a1, gamma)
    K[0, 1] = rbf_kernel(a1, a2, gamma)
    K[1, 0] = rbf_kernel(a2, a1, gamma)
    K[1, 1] = rbf_kernel(a2, a2, gamma)

    # 使用特征值分解来近似映射
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = eigvals[::-1]  # 从大到小排序
    eigvecs = eigvecs[:, ::-1]

    # 选择前6个最大的特征值和对应的特征向量
    eigvals = eigvals[:6]
    eigvecs = eigvecs[:, :6]

    # 映射到6维空间
    mapped_a1 = np.sqrt(eigvals) * eigvecs[0, :]
    mapped_a2 = np.sqrt(eigvals) * eigvecs[1, :]

    return mapped_a1, mapped_a2

# 设置gamma值
gamma = 0.5

# 映射数据到6维空间
mapped_a1, mapped_a2 = map_to_6d(a1, a2, gamma)

print("原始数据 a1:", a1)
print("原始数据 a2:", a2)
print("映射后的数据 a1:", mapped_a1)
print("映射后的数据 a2:", mapped_a2)