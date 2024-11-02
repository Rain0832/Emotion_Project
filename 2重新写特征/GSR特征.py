import matplotlib.pyplot as plt
import numpy as np  # 导入NumPy库
import pandas as pd
from scipy.signal import argrelextrema  # 导入查找局部极小值的函数
import matplotlib.pyplot as plt

signall = pd.read_excel(r"E:\2大学项目\2重新写特征\880原GSR数据（带行列索引）.xlsx",
                        index_col=0, usecols=range(0, 8065), skiprows=1, nrows=1)

plt.plot( np.array(signall).flatten())  # 使用 iloc 来选择第 1 列
plt.show()
# print(np.array(signall))
# signall = signall.iloc[880,:]
# signall=[1.000002,1.000000,1.000002,1.000002,1.000003,1.000004,1.000004,1.000002,1.000002,1.000005,1.00002]
# print(signall)

# 1. 计算导数
derivative = np.diff(signall)  # 计算信号的一阶导数
# 2. 计算导数的平均值
average_derivative = np.mean(derivative)
# 3. 计算导数中负样本的百分比
negative_samples = derivative[derivative < 0]  # 提取负样本
negative_percent_in_deriv = len(negative_samples) / len(derivative) * 100  # 计算百分比


# 4. 查找局部极小值的数量
# 首先将 signall 转换为 NumPy 数组

def moving_average(signal, window_size):
    # 使用卷积计算移动平均
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')


signall = np.array(signall).flatten()
plt.plot(signall)
plt.show()
print(signall)
# 应用移动平均，窗口大小为3
smoothed_signall = moving_average(signall, 700)
signall_np = np.array(smoothed_signall).flatten()

# print(signall_np)
local_minima_indices = argrelextrema(signall_np, np.less, order=400)  # 查找局部极小值的索引
local_minima_indices = local_minima_indices[0]
print("ssss", local_minima_indices)
number_of_local_minima = len(local_minima_indices)  # 计算数量

# 输出结果
# print(f"Average Derivative: {average_derivative}")
# print(f"Negative Samples Percentage: {negative_percent_in_deriv:.2f}%")
print(f"Number of Local Minima: {number_of_local_minima}")
