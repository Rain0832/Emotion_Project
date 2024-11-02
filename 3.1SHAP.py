import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt


"""
此程序是为了画各个特征的SHAP主图
SHAP： 事后解释
甲特征的边际贡献：加入特征甲后，相比未加入时，甲带来的贡献
甲的SHAP值：所有特征组合下，甲的所有边际贡献的平均值
一个样本中各个特征的SHAP值+基线值=模型的最终预测值
"""
# # 读取数据
# df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
#                    index_col=0, usecols=range(0, 52), nrows=None)
# new_df = df.iloc[:, 0:51]
# X = new_df.iloc[:, :-1]
# y = df.iloc[:, -1]

# 读取数据
df = pd.read_excel(r"E:\2大学项目\选择合适的特征\start to go\2\标准化数据_方差均值5（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 7), nrows=None)
new_df = df.iloc[:, 0:7]
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]

# print(new_df)#[892 rows 前6个特征
# 分割数据集
X_train, X_test, y_train, y_test = X[:700], X[700:], y[:700], y[700:]

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
# 解决中文乱码
# plt.rcParams["font.sans-serif"]=["SimSun"]
# plt.rcParams["font.family"]="sans-serif"
#负号
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
# 确保 X_test 是一个带有特征名称的 DataFrame
feature_names = ["特征9","特征33","特征36","特征38","特征39"]
X_test_df = pd.DataFrame(X_test, columns=feature_names)
# print(X_test_df)
"""
explainer_new = 是背景数据，作为计算shap的参考，亦即在原始情况下的表现)
shap_values = explainer_new.shap_values(X_test_df)是对测试集进行组合等，与背景数据对应的情况相比较计算SHAP值
"""
explainer_new = shap.KernelExplainer(svm_model.predict_proba, X_train)
shap_values = explainer_new.shap_values(X_test_df)
shap.summary_plot(shap_values[:, :, 1], X_test_df,max_display=5)

"""
Using 200 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
  2%|▏         | 4/192 [00:27<21:38,  6.91s/it]
Using 800 background data samples
 1%|          | 1/92 [01:51<2:48:50, 111.33s/it]
可以训练集400，测试集100（重要的是深入了解shap原理
此图的解释
每个特征值对结果1的预测程度，理想状态：蓝与红分离，即值小的可能对1的贡献小（反映为SHAP值小），值大的可能贡献大，或其他可能
SHAP值：正的：促进结果1，负的抑制

基线值（Base Value）：模型在不考虑任何特征时的预测值，通常是训练集的平均预测值。
特征贡献（Feature Contributions）：每个特征对模型预测的影响，表现为水平条形图，红色表示正向贡献（推动预测值增加），蓝色表示负向贡献（推动预测值减少）。
特征值（Feature Values）：图中显示了每个特征的实际值，这有助于理解为什么某个特征会有正向或负向的贡献。
"""
""""
Using 300 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
100%|██████████| 192/192 [1:51:05<00:00, 34.71s/it]
"""