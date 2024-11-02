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
画SHAP全局条形图
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
new_df = df.iloc[:, 0:51]  #new_df = df.iloc[:, 0:7] 前6特征
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(new_df)#[892 rows 前6个特征
# 分割数据集
X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]

shap.initjs()
# 限制训练集大小
X_train = X_train[:100]
y_train = y_train[:100]
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train)  # 应该传入训练集
shap_values = explainer(X_train)
print("shap_values[0]", shap_values[0].shape)  # (6, 2)
print("shap_values[0,:,1]", shap_values[0, :, 1])  # shap_values[0,:,1] (6,)
print("shap_values", shap_values.shape)  # shap_values (30, 6, 2)
print("explainer.expected_value[1]", explainer.expected_value[1])
print("explainer.expected_value", explainer.expected_value)
# 解决中文乱码
# plt.rcParams["font.sans-serif"]=["SimHei"]
# plt.rcParams["font.family"]="sans-serif"
# 解决负号无法显示的问题
plt.rcParams['axes.unicode_minus'] =False
plt.rcParams['axes.unicode_minus'] =False
shap.plots.bar(shap_values[:,:,1], max_display=50)#shap_values[0,:,1]取第一个样本，标签是第二类1
plt.show()
"""
解释图形：全局条形图：
先用训练集许多样本计算SHAP值，选择使用样本
计算每个特征的 SHAP 值的绝对值的平均值，将其画出图展示
"""



