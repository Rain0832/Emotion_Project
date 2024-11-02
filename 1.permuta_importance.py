import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
permutation importance
对某一特征，看它有没有用，将特征的数值打乱（不同样本间调换位置）或加噪声，
带入模型训练，看模型的错误率，与原始正常模型的错误率比较
"""
# 读取数据（假设数据存储在Excel文件中）
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)  # probability=True：启用概率估计，用于后续的SHAP值计算。
# 预测
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import eli5
from eli5 import permutation_importance

perm = permutation_importance.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=X_test.columns.tolist)

