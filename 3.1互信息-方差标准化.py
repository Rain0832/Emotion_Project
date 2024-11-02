import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectKBest, chi2
from matplotlib.colors import Normalize
from matplotlib.pyplot import get_cmap
from matplotlib import cm
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)

new_df = df.iloc[:, 0:51]
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
from sklearn.feature_selection import mutual_info_classif as mic
result = mic(X,y)
print("每个特征的互信息值",result)#result不等于0则表示有信息含量
k = result.shape[0]-sum(result<=0) #result.shape[0]=50， 计算具有非零互信息的特征数量：
print("保存特征数量：",k)

# 获取互信息值的索引，并降序排序
sorted_indices = result.argsort()[::-1]  # 使用 argsort 获取索引，然后逆序
# 按互信息值降序排序特征
sorted_features = [X.columns[i] for i in sorted_indices]
# 打印排序后的特征及其互信息值
for feature, index in zip(sorted_features, sorted_indices):
    print(f"Feature: {feature}, Mutual Information: {result[index]:.9f}")


# plt.bar(range(1, 51), result,tick_label=range(1,51))
# plt.xlabel('Feature Index')
# plt.ylabel('Mutual Information')
# plt.title('Mutual Information for Each Feature')
# plt.show()
#彩色的特征的互信息值
# 创建颜色渐变
norm = plt.Normalize(vmin=min(result), vmax=max(result))
cmap = plt.get_cmap('viridis_r')
colors = cmap(norm(result))

# 绘制柱状图
fig, ax = plt.subplots()
bars = ax.bar(range(1, len(result) + 1), result, tick_label=range(1, len(result) + 1), color=colors)  # 从1开始计数

# 在柱子的顶端标注y值
for bar in bars:
    yval = bar.get_height()
    if abs(yval - 0) <= 0.0001:
        label = "0"
    else:
        label = f"{yval:.4f}"
    ax.text(bar.get_x() + bar.get_width() / 2, yval, label, ha='center', va='bottom')

# 设置Y轴刻度为原始数据的范围，并根据需要设置合适的刻度数量
ax.set_ylim(0, max(result) * 1.1)  # 扩展一点范围以便所有条形都能被看清楚

# 设置X轴和Y轴的刻度
ax.set_xticks(range(1, len(result) + 1))  # X轴刻度从1开始
ax.set_yticks(np.linspace(0, max(result), 11))  # 根据数据范围生成Y轴刻度

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Mutual Information')

# 设置标签和标题
ax.set_xlabel('Feature Index')
ax.set_ylabel('Mutual Information')
ax.set_title('Mutual Information for Each Feature')

plt.show()