import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt
import matplotlib

"""
此程序是为了画各个特征的SHAP主图
SHAP： 事后解释
甲特征的边际贡献：加入特征甲后，相比未加入时，甲带来的贡献
甲的SHAP值：所有特征组合下，甲的所有边际贡献的平均值
一个样本中各个特征的SHAP值+基线值=模型的最终预测值
"""

df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# print(X_test)

# print(new_df)#[892 rows 前6个特征

# 限制训练集大小
# X_train = X_train[:500]
# y_train = y_train[:500]
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
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 确保 X_test 是一个带有特征名称的 DataFrame
feature_names = ["特征1","特征2","特征3","特征4","特征9"]
X_test_df = pd.DataFrame(X_test, columns=feature_names)
# print(X_test_df)
"""
explainer_new = 是背景数据，作为计算shap的参考，亦即在原始情况下的表现)
shap_values = explainer_new.shap_values(X_test_df)是对测试集进行组合等，与背景数据对应的情况相比较计算SHAP值
"""
explainer_new = shap.KernelExplainer(svm_model.predict_proba, X_train)
shap_values = explainer_new.shap_values(X_test_df)
shap.summary_plot(shap_values[:, :, 1], X_test_df,max_display=4)#max_display最多显示几个特征

