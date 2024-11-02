import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import matplotlib

""""
使用方差标准化的数据，打印
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
# 假设 '二分类' 是目标列，如果不是，请根据实际情况调整
y = df[['二分类']].values.ravel()  # 确保y是一维数组
X = df.iloc[:, 0:-1]
# 初始化一个字典来存储每个特征的F值和p值
feature_scores = {}

# 计算每个特征的F值和p值
for i in range(0, 50):
    feature = df.iloc[:, i]
    F, p = f_classif(feature.values.reshape(-1, 1), y)
    feature_scores[df.columns[i]] = (F[0], p[0])  # 提取数组的第一个元素

# 将结果转换为DataFrame以便于可视化
results_df = pd.DataFrame.from_dict(feature_scores, orient='index', columns=['F', 'p'])
# print("results_df", results_df)
# 按照F值降序排序
sorted_by_F = results_df.sort_values(by='F', ascending=False)
print("Sorted by F value:")
for index, row in sorted_by_F.iterrows():
    print(f"Feature: {index}, F value: {row['F']:.3f}")

# 按照p值升序排序
sorted_by_p = results_df.sort_values(by='p', ascending=True)
print("Sorted by p value:")
for index, row in sorted_by_p.iterrows():
    print(f"Feature: {index}, p value: {row['p']:.4f}")

from sklearn.feature_selection import f_classif, SelectKBest  #

KB_CF = SelectKBest(f_classif, k=2)  # k=2，选取两个最重要特征
KB_CF.fit(X, y)  # SelectKBest(k=2)
selected_features = KB_CF.get_support()  # 对每一个特征，展示true 还是false
print("the Trues are:\n", selected_features)

print("KB_CF.scores_\n", KB_CF.scores_)
print("KB_CF.pvalues_\n", KB_CF.pvalues_)
