import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

"""
置换重要性:（Permutation Importance）：通过随机打乱某个特征的值，观察模型性能的变化来评估特征重要性。
permutation importance
对某一特征，看它有没有用，将特征的数值打乱（不同样本间调换位置）或加噪声，带入模型训练，看模型的错误率，与原始正常模型的错误率比较

"""
# X.shape = (892, 50)
# df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
#                    index_col=0, usecols=range(0, 52), nrows=None)
df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM模型
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# 计算置换重要性
result = permutation_importance(svm_model, X_test, y_test, n_repeats=30, random_state=42)
# 获取特征重要性
feature_importances = result.importances_mean
print("feature importance:", feature_importances)

# 获取特征名称
feature_names = df.columns[:-1].tolist()
# 对特征重要性进行排序（按降序）
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]
# 打印排序后的特征重要性
for name, importance in zip(sorted_feature_names, sorted_feature_importances):
    print(f"{name}: {importance}")

import matplotlib.pyplot as plt
import matplotlib
# 获取特征名称
feature_names = df.columns[:-1].tolist()
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False
# 绘制特征重要性
plt.figure(figsize=(10, 15))
plt.barh(range(len(feature_importances)), feature_importances, align='center')
plt.yticks(range(len(feature_importances)), feature_names)
plt.xlabel('Permutation Importance')
plt.title('Feature Importance')
plt.show()
#彩图
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
# 获取特征名称
feature_names = df.columns[:-1].tolist()

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# # 创建一个颜色映射，例如从绿色到黄色到红色
# cmap = plt.get_cmap('RdYlGn')
# 定义颜色段
cmap_name = 'GreenOrangeRed'
cmap = LinearSegmentedColormap.from_list(cmap_name, ['green', 'orange', 'red'])

# 归一化特征重要性值
norm = matplotlib.colors.Normalize(vmin=min(feature_importances), vmax=max(feature_importances))

# 绘制特征重要性
fig, ax = plt.subplots(figsize=(10, 15))
bars = ax.barh(range(len(feature_importances)), feature_importances, align='center')

# 设置每个柱子的颜色
for bar, importance in zip(bars, feature_importances):
    color = cmap(norm(importance))
    bar.set_color(color)

ax.set_yticks(range(len(feature_importances)))
ax.set_yticklabels(feature_names)
ax.set_xlabel('Permutation Importance')
ax.set_title('Feature Importance')

# 添加颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Importance')

plt.show()
# """
# 负值可能表明该特征在原始数据中对模型的预测贡献不大，
# 但随机打乱后，由于数据的随机性，模型的预测准确性反而提高了。
# 这种情况在小数据集中更为常见，因为小数据集中存在更多的随机性或噪声
# """