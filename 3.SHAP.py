import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

"""
对一个样本：
打印各个特征的SHAP值，画SHAP力图
特征数，6-》50
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
new_df = df.iloc[:, 0:7]  #new_df = df.iloc[:, 0:7] 前6特征
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(new_df)#[892 rows 前6个特征
# 分割数据集
X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]

# print(X_train)#前30样本

"""
先对一个样本的6个特征进行SHAP
"""
# row_selected = 4
# single_line = X_train.iloc[row_selected]  #第5个样本，包含6个特征[ 0.45465947  0.42924142  0.27591596 -0.11569815  0.50272503]
# single_line_array = np.array(single_line).reshape(1, -1)
# y_single = y.iloc[row_selected]
# y_single_array = np.array([y_single])
# print(y_single)
# print(single_line)#第5个样本，包含6个特征
# # print("data_for_predict_array",data_for_predict_array)#[[ 0.45465947  0.42924142  0.27591596 -0.11569815  0.50272503]]
#
# # 训练SVM模型
# svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
# svm_model.fit(X_train, y_train)
# # 预测
# y_pred = svm_model.predict(X_test)
#
# print("0,1概率", svm_model.predict_proba(single_line_array))  #[[0.21332197 0.78667803]]
#
# explainer = shap.KernelExplainer(svm_model.predict_proba, single_line_array)
# shap_value = explainer.shap_values(single_line_array)
# print("shap_value[0]",shap_value[0])
"""
shap_valve三维数组的含义：(n_samples, n_features, n_classes)，其中：
n_samples是评估SHAP值的数据集中的样本数。
n_features是数据集中的特征数。
n_classes是模型输出的类别数。对于二分类问题，n_classes为2；
"""
# print("shap_value", shap_value.shape)  #shap_value (1, 6, 2)

shap.initjs()
# 限制训练集大小
X_train = X_train[:30]
y_train = y_train[:30]
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)
explainer_new = shap.KernelExplainer(svm_model.predict_proba, X_train)  # 应该传入训练集
shap_values = explainer_new.shap_values(X_test)
print("shap_values[0]", shap_values[0].shape)  # (6, 2)
print("shap_values[0,:,1]", shap_values[0, :, 1])  # shap_values[0,:,1] (6,)
print("shap_values", shap_values.shape)  # shap_values (292, 6, 2)
print("explainer_new.expected_value[1]", explainer_new.expected_value[1])
print("explainer_new.expected_value", explainer_new.expected_value)
print("X_test[:1]", X_test[:1])
print("X_test[:1]", X_test[:1].shape)
# 设置默认字体为 SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

shap.force_plot(explainer_new.expected_value[1],shap_values[0,:,1] ,X_test[:1])
shap.force_plot(
    base_value=explainer_new.expected_value[1],  # 第二个类别的期望值
    shap_values=shap_values[0, :, 1],  # 第一个样本的 SHAP 值（第二个类别）
    features=X_test[:1],  # 第一个样本的特征值
    show=False, matplotlib=True)
plt.show()
"""
SHAP力图
"""
