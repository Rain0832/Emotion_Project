import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import seaborn as sns
from sklearn.model_selection import GridSearchCV
"""
方差过滤
"""
# 读取数据
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_最小最大值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
new_df = df.iloc[:, 0:51]
X = new_df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(new_df)#[892 rows 前6个特征
# 分割数据集
# X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# # 限制训练集大小
# X_train = X_train[:100]
# y_train = y_train[:100]
# print(X_train)#前30样本
# 训练SVM模型
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)
# # 预测
y_pred = svm_model.predict(X_test)
print("raw:", accuracy_score(y_test, y_pred), "recall:", recall_score(y_test, y_pred), "f1", f1_score(y_test, y_pred))
# y_pred = svm_model.predict(X_train)
# print("raw:", accuracy_score(y_train, y_pred),"recall:",recall_score(y_train y_pred), "f1", f1_score(y_train, y_pred))

#进行方差过滤
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from sklearn.tree import DecisionTreeClassifier
mid_val = np.median(X.var().values)  # 各个特征方差的中位数
selector = VarianceThreshold(mid_val)  # 中位数作为阈值
X_filter = selector.fit_transform(X)  # 保留一半的特征

# 计算各个特征的方差，使用np.var得到方差数组
variances = np.var(X, axis=0)
# 获取特征名和方差值的列表，然后按方差降序排列
features_with_var = sorted(zip(X.columns, variances), key=lambda x: x[1], reverse=True)

# 打印出保留的特征及其方差
for feature_name, variance in features_with_var:
    if variance > mid_val:
        print(f"Feature {feature_name} with variance {variance} is retained")
# print("x shape", X.shape)
# print("x ",X_filter)
X_train, X_test, y_train, y_test = train_test_split(X_filter, y, test_size=0.3, random_state=1, stratify=y)

# # 构造参数网格
# param_grid = {
#     'C': np.arange(0.1, 10, 0.1),  # 正则化参数
#     'kernel': ['rbf'],  # 核函数类型
#     'gamma': np.arange(0.00001, 0.001, 0.0001)  # 核函数的系数
# }
# # 实例化网格搜索评估器
# grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# #cv 交叉验证的折数，分为几组，n_jobs=-1：使用所有可用的CPU核心。verbose=2：控制输出的详细程度，2表示更详细的输出。
# grid_search_svm.fit(X_train, y_train)  # 在训练集上，执行网格搜索
# # 打印最佳参数和最佳分数
# print("Best parameters found:",grid_search_svm.best_params_)
# print("Best cross-validation score:", grid_search_svm.best_score_)
# # 获取最佳SVM模型
# best_svm = grid_search_svm.best_estimator_
# print("best svm is", best_svm)
# """
# SVC(C=np.float64(1e-05), decision_function_shape='ovo', gamma=np.float64(1e-05),probability=True)
# """
# 训练SVM模型
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)
# # 预测
y_pred = svm_model.predict(X_test)
print("new:", accuracy_score(y_test, y_pred),"recall:",recall_score(y_test, y_pred), "f1", f1_score(y_test, y_pred))
# y_pred = svm_model.predict(X_train)
# print("raw:", accuracy_score(y_train, y_pred),"recall:",recall_score(y_train, y_pred), "f1", f1_score(y_train, y_pred))



result = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵,只对测试集
def show_confusion_matrix(result):
    plt.figure(figsize=(8, 6))
    sns.heatmap(result,
                annot=True,  # 在热图上显示值
                linewidths=.7,
                fmt='d',  # 显示整数格式
                cmap='YlOrRd',  # 从黄色到红色的渐变
                xticklabels=['Predicted: 0', 'Predicted: 1'],
                yticklabels=['Actual: 0', 'Actual: 1'],
                linecolor='green')  # 设置边框颜色为绿色

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
show_confusion_matrix(result)
