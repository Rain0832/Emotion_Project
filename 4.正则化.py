import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
"""
L1正则化只对线性SVC可用；
非线性的自带是L2正则化，但是同样卡在feature importance不知道
程序：使用真阴率网格搜索，线性SVC，选特征
"""
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
# svm_model.fit(X_train, y_train)  # probability=True：启用概率估计，用于后续的SHAP值计算。

# selector = SelectFromModel(svm_model, threshold=0, prefit=True)
# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)
# # 检查被选择的特征数量
# print(f"Number of features selected: {X_train_selected.shape[1]}")


best_f1_score = 0
best_svc = None
best_C = None

# for i in np.arange(0.01, 10, 0.1):
#     # 使用L1正则化创建LinearSVC模型
#     l1_svc = LinearSVC(penalty='l1', dual=False, C=i)
#     l1_svc.fit(X_train, y_train)
#     # 预测测试集
#     l1_predictions = l1_svc.predict(X_test)
#     l1_f1_score = f1_score(y_test, l1_predictions)
#
#     if l1_f1_score > best_f1_score:
#         best_f1_score = l1_f1_score
#         best_svc = l1_svc
#         best_C = i
#         l1_accuracy = accuracy_score(y_test, l1_predictions)
#         l1_recall = recall_score(y_test, l1_predictions)
#
# print(f"Best C: {best_C}")
# print(f"Accuracy: {l1_accuracy}, Recall: {l1_recall}, F1: {best_f1_score}")

best_specificity = 0
for i in np.arange(0.01, 10, 0.1):
    # 使用L1正则化创建LinearSVC模型
    l1_svc = LinearSVC(penalty='l1', dual=False, C=i)
    l1_svc.fit(X_train, y_train)
    # 预测测试集
    l1_predictions = l1_svc.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, l1_predictions)
    tn, fp, fn, tp = cm.ravel()

    # 计算真阴率（Specificity）
    specificity = tn / (tn + fp)

    if specificity > best_specificity:
        best_specificity = specificity
        best_svc = l1_svc
        best_C = i
        best_cm =cm
        l1_accuracy = accuracy_score(y_test, l1_predictions)
        l1_recall = recall_score(y_test, l1_predictions)
        l1_f1=f1_score(y_test, l1_predictions)

print(f"Best C: {best_C}")
print(f"Accuracy: {l1_accuracy}, Recall: {l1_recall}, Specificity: {best_specificity},f1:{l1_f1}")
print("confusion  metrix\n",best_cm)
# 获取权重
weights = best_svc.coef_

# 可视化权重
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), weights[0], alpha=0.8)
plt.title('Feature Weights from LinearSVC with L1 Regularization')
plt.xlabel('Feature Index')
plt.ylabel('Weight')
plt.xticks(range(X_train.shape[1]), [f"Feature {i+1}" for i in range(X_train.shape[1])], rotation=90)
plt.show()

# 可视化被选中的特征
non_zero_features = np.sum(weights != 0, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), non_zero_features, alpha=0.8)
plt.title('Features Selected by LinearSVC with L1 Regularization')
plt.xlabel('Feature Index')
plt.ylabel('Selected (Non-zero Weight)')
plt.xticks(range(X_train.shape[1]), [f"Feature {i+1}" for i in range(X_train.shape[1])], rotation=90)
plt.show()