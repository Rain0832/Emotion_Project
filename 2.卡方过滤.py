import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectKBest, chi2

# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_最小最大值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)

new_df = df.iloc[:, 0:51]
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]

# svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
# print("原始交叉验证5组的结果",cross_val_score(svm_model, X, y, cv=5))

X_chi = SelectKBest(chi2, k=25).fit_transform(X, y)  # 进行卡方过滤，k=保留卡方最大的特征数
# print(X_chi.shape)# (892, 25)
# print("卡方过滤后交叉验证5组的结果",cross_val_score(svm_model, X_chi, y, cv=5))

selector = SelectKBest(chi2, k=25)
selector.fit(X, y)  # 确保计算得分
# 获取每个特征的卡方检验值
chi_scores = selector.scores_
"""
进行降序排序
"""
# 获取排序后的卡方得分和对应的特征索引，降序排序
sorted_indices = np.argsort(-chi_scores)  # 使用负号进行降序排序

# 按卡方得分降序排序特征
sorted_chi_scores = [-chi_scores[i] for i in sorted_indices]  # 使用负号转换为正数
sorted_features = [X.columns[i] for i in sorted_indices]

# 打印按卡方得分排序的特征及其得分
for feature, score in zip(sorted_features, sorted_chi_scores):
    print(f"Feature: {feature}, Chi-Square Score: {-score:.3f}")


# 绘制卡方检验值的柱状图,卡方值越大：表示实际观测频数和期望频数之间的差异越大，说明特征和目标变量之间的关联越强。
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(chi_scores)), chi_scores, tick_label=X.columns)
# 在每个条形上标注Y轴的值，保留3位小数，如果是0，则写0
for bar in bars:
    yval = bar.get_height()
    if abs(yval - 0)<=0.001:
        label = "0"
    else:
        label = f"{yval:.3f}"
    plt.text(bar.get_x() + bar.get_width() / 2, yval, label, ha='center', va='bottom')
plt.xlabel('Feature Index')
plt.ylabel('Chi-Square Score')
plt.title('Chi-Square Scores for Each Feature')
plt.xticks(rotation=90)  # 旋转X轴标签以便更容易阅读

plt.show()

#删特征后得分随保留特征的图
# score = []
# # 为了确定要筛选几个特征，绘出交叉验证得分随保留特征数的变化
# for i in range(50, 0, -1):
#     X_chi_new = SelectKBest(chi2, k=i).fit_transform(X, y)
#     once = cross_val_score(svm_model, X_chi_new, y, cv=5).mean()
#     score.append(once)
#
# # 绘图,
# fig, ax = plt.subplots()
# ax.plot(range(50, 0, -1), score, marker='o')
# # 添加文本标签和垂直线
# for i, txt in enumerate(score):
#     ax.annotate(f"{txt:.2f}", (50 - i, score[i]), textcoords="offset points", xytext=(0,10), ha='center')
#     ax.axvline(x=50 - i, color='gray', linestyle='--', alpha=0.5)
#     ax.text(50 - i, ax.get_ylim()[0], f"{50 - i}", va='bottom', ha='center', fontsize=8, rotation=45)
#
# plt.xlabel('Number of Features')
# plt.ylabel('Cross-Validation Score')
# plt.title('chi-square test ** throwing features ** SVM ')
# plt.show()
