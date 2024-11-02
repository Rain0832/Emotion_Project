import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
"""
画某一个特征的SHAP值与特征值的散点图
以及另一特征对此的交互
用处不大
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
new_df = df.iloc[:, 0:51]  #new_df = df.iloc[:, 0:51]全部特征
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(new_df)#[892 rows 前6个特征
# 分割数据集
X_train, X_test, y_train, y_test = X[:800], X[880:], y[:800], y[880:]

# # 限制训练集大小
# X_train = X_train[:100]
# y_train = y_train[:100]
# print(X_train)#前30样本
# 训练SVM模型
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)
# 预测
y_pred = svm_model.predict(X_test)

"""
shap_valve三维数组的含义：(n_samples, n_features, n_classes)，其中：
n_samples是评估SHAP值的数据集中的样本数。
n_features是数据集中的特征数。
n_classes是模型输出的类别数。对于二分类问题，n_classes为2；
"""
#看单个特征的SHAP

explainer_new = shap.KernelExplainer(svm_model.predict_proba, X_test)
explanation = explainer_new(X_test)
print(explanation)
shap.plots.scatter(explanation[:, "特征1", 1])  #简单依赖散点图显示单个特征对模型所做的预测的影响

#交互图
shap.plots.scatter(explanation[:, "特征1",1], color=explanation[:, "特征2",1])
"""
此图的解释
每个特征值对结果1的预测程度，理想状态：蓝与红分离，即值小的可能对1的贡献小（反映为SHAP值小），值大的可能贡献大，或其他可能
SHAP值：正的：促进结果1，负的抑制
"""
