import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, make_scorer
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
best_f1_score = 0
best_svc = None
best_C = None
best_specificity = 0


# 自定义真阴率（Specificity）的评分函数
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)##真阴/（真阴+假阳）-》所有阴性



# 创建自定义评分器
specificity_scorer = make_scorer(specificity_score, greater_is_better=True)

# 使用L1正则化创建LinearSVC模型
l1_svc = LinearSVC(penalty='l1', dual=False)
# 定义参数网格
param_grid = {'C': np.arange(0.01, 10, 0.1)}
# 使用网格搜索和交叉验证
grid_search = GridSearchCV(l1_svc, param_grid, scoring=specificity_scorer, cv=5)
grid_search.fit(X_train, y_train)
print("cv best specificity_score:", grid_search.best_score_)
# 获取最佳模型和参数
best_svc = grid_search.best_estimator_
best_C = grid_search.best_params_['C']
print( "*****", cross_val_score(best_svc, X_train, y_train, cv=5, scoring=specificity_scorer))
# 预测测试集
l1_predictions = best_svc.predict(X_test)

# 计算评估指标
l1_accuracy = accuracy_score(y_test, l1_predictions)
l1_recall = recall_score(y_test, l1_predictions)
l1_f1 = f1_score(y_test, l1_predictions)
best_specificity = specificity_score(y_test, l1_predictions)
best_cm = confusion_matrix(y_test, l1_predictions)

print(f"Best C: {best_C}")
print(f"Accuracy: {l1_accuracy}, Recall: {l1_recall}, Specificity: {best_specificity}, F1: {l1_f1}")
print("Confusion Matrix:\n", best_cm)

# 获取权重
weights = best_svc.coef_

# 可视化权重
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), weights[0], alpha=0.8)
plt.title('Feature Weights from LinearSVC with L1 Regularization')
plt.xlabel('Feature Index')
plt.ylabel('Weight')
plt.xticks(range(X_train.shape[1]), [f"Feature {i + 1}" for i in range(X_train.shape[1])], rotation=90)
plt.show()

# 可视化被选中的特征
non_zero_features = np.sum(weights != 0, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), non_zero_features, alpha=0.8)
plt.title('Features Selected by LinearSVC with L1 Regularization')
plt.xlabel('Feature Index')
plt.ylabel('Selected (Non-zero Weight)')
plt.xticks(range(X_train.shape[1]), [f"Feature {i + 1}" for i in range(X_train.shape[1])], rotation=90)
plt.show()
