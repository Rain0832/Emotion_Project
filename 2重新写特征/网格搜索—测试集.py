import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
"""
使用f1为指标，交叉验证
"""
df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

def specificity_scoring(estimator, X, y_true):
    y_pred = estimator.predict(X)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp = conf_matrix[0, 0], conf_matrix[0, 1]  # 混淆矩阵的真阴例和假正例
    return tn / (tn + fp) if (tn + fp) > 0 else 0


# 定义SVM模型
svm_model = SVC(decision_function_shape='ovo')

# 构造参数网格
param_grid = {
    'C': np.arange(9.9, 10.1, 0.1),  # 正则化参数
    'kernel': ['rbf'],  # 核函数类型
    'gamma': np.arange(8.9, 10.5, 0.01)  # 核函数的系数
}
# 实例化网格搜索评估器，本身就是交叉验证的
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=2)
#cv 交叉验证的折数，分为几组，n_jobs=-1：使用所有可用的CPU核心。verbose=2：控制输出的详细程度，2表示更详细的输出。
grid_search_svm.fit(X_train, y_train)  # 在训练集上，执行网格搜索scoring=tnr_scorer
# 打印最佳参数和最佳分数
print("Best parameters found:", grid_search_svm.best_params_)
print("Best cross-validation score:", grid_search_svm.best_score_)
# 获取最佳SVM模型
best_svm = grid_search_svm.best_estimator_
print("best svm is", best_svm)  #
best_svm.fit(X_train,y_train)
y_pred = best_svm.predict(X_test) # 预测的y值
cross_validation = cross_val_score(best_svm, X_test, y_test, cv=5, scoring="f1")  # cv是分成几份
print("test集交叉验证结果真阴率:", cross_validation)

accu = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("accuracy=", accu, " ; recall=", recall, " ; f1=", f1)

result = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
print("混淆矩阵\n", result)
tn, fp, fn, tp = result[0, 0], result[0, 1], result[1, 0], result[1, 1]
print("tesy集真阴率", tn / (tn + fp))

"""
Best parameters found: {'C': np.float64(9.0), 'gamma': np.float64(9.1), 'kernel': 'rbf'}
Best cross-validation score: 0.6885120431517288
best svm is SVC(C=np.float64(9.0), decision_function_shape='ovo', gamma=np.float64(9.1))
test集交叉验证结果真阴率: [0.69230769 0.7012987  0.69333333 0.68421053 0.71232877]
accuracy= 0.5597014925373134  ; recall= 0.8280254777070064  ; f1= 0.6878306878306878
混淆矩阵
 [[ 20  91]
 [ 27 130]]
训练集真阴率 0.18018018018018017
"""