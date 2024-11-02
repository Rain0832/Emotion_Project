import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import matplotlib
""""
使用方差标准化的数据,画图
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
# 假设 '二分类' 是目标列，如果不是，请根据实际情况调整
y = df[['二分类']].values.ravel()  # 确保y是一维数组
# 初始化一个字典来存储每个特征的F值和p值
feature_scores = {}

# 计算每个特征的F值和p值
for i in range(0,50):
    feature = df.iloc[:, i]
    F, p = f_classif(feature.values.reshape(-1, 1), y)
    feature_scores[df.columns[i]] = (F, p)

# 将结果转换为DataFrame以便于可视化
results_df = pd.DataFrame.from_dict(feature_scores, orient='index', columns=['F', 'p'])
print("results_df",results_df)
F_values = results_df['F'].values
F_flattened= np.concatenate(F_values)
p_values = results_df['p'].values
p_flattened = np.concatenate(p_values)###将每个特征的F与p值降序输出特征名称与值





# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 可视化F值
plt.figure(figsize=(10, 6))
bars_F=plt.barh(range(len(results_df)), F_flattened, color='skyblue')
plt.xlabel('F Value')
# 设置Y轴的刻度位置和标签
plt.yticks(range(len(F_flattened)), [f"特征{i+1}" for i in range(len(F_flattened))])
plt.title('F Values for Each Feature')
plt.gca().invert_yaxis()  # 反转y轴，使得最大的F值在顶部
# 在每个条形上标注 F 值
plt.show()

# 可视化p值
plt.figure(figsize=(10, 6))
plt.barh(range(len(results_df)), p_flattened, color='lightcoral')
plt.yticks(range(len(p_flattened)), [f"特征{i+1}" for i in range(len(p_flattened))])
plt.xlabel('p Value')
plt.title('p Values for Each Feature')
plt.gca().invert_yaxis()  # 反转y轴，使得最小的p值在顶部
plt.show()