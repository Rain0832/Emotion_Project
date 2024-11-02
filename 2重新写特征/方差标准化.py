import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
df = pd.read_excel(r"E:\2大学项目\2重新写特征\2重新写特征\筛选后特征+标签3（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 5), nrows=None)
"""
index_col=0: 将第一列作为索引列。
usecols=range(0, 52): 读取第0到第51列（共52列）。
nrows=None: 读取所有行。
"""

df.columns = ['特征' + str(i + 1) for i in range(4)]  # 将列索引重命名为“特征1”, “特征2”, “特征3”等， 到特征51
# 特征和目标变量分离
X = df.iloc[:, :-1].values  # 特征 (892, 50)     iloc: 基于整数位置的索引。[:, :-1]: 所有行，除最后一列外的所有列。
y = df.iloc[:, -1].values  # 目标变量 (892,)     所有行，最后一列。

# 定义一个标准缩放器并计算均值、标准差，然后进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 先调用fit计算均值和标准差，然后调用transform进行标准化
"""
StandardScaler()
均值：对每个特征求均值，即对每列求均值
去均值：每个特征的值减去对应特征的均值
标准差：去均值后平方和，然后除以总值的数量，最后开根号
标准分数：去均值、除以标准差
1、中心化：去均值，将整体数据平移，中心为(0,0)
2、缩放：标准分数，进行缩放
标准化：去均值、除以标准差。将数据的分布转为正态分布。每个特征的值 均值=0、方差=1
    目的：
        将特征表现为标准正态分布数据。
        如果某个特征的方差比其他特征大几个数量级，那么它就会在学习算法中占据主导位置，导致学习器不能从其他特征中学习，从而降低精度。
		加快梯度下降求解的速度。
"""
# 创建包含标准化特征的DataFrame
normalized_df = pd.DataFrame(X_scaled, index=df.index, columns=df.columns[:-1])
           # index=df.index: 使用原始DataFrame的行索引。 columns=df.columns[:-1]: 使用原始DataFrame的列名（除最后一列）。

# 将目标变量'二分类'添加到标准化特征的DataFrame中
normalized_df['二分类'] = y

resultPath1 = r"E:\2大学项目\2重新写特征\2重新写特征\\标准化数据_方差均值1（带行列索引）.xlsx"
normalized_df.to_excel(resultPath1, sheet_name="标准化数据（带行列索引）", index=True)  # index=True: 保存索引列。
