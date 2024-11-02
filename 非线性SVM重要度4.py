import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel(r"E:\2大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
columns_to_drop = ['二分类']
y = df[columns_to_drop]
y = y.values.flatten()  # 一维列表
X = df.iloc[:, :50]  # (892, 49)
X_train, y_train = X[:600], y[:600]  # 前600行数据(0~599)
X_test, y_test = X[600:], y[600:]  # 600行以及之后的(600~891)
# 特征标准化
def normalize_features(df):
    normalized_df = pd.DataFrame()
    for i in range(0, 50):  #一列一列标准化
        col = df.iloc[:, i]
        min_col = col.min()
        max_col = col.max()
        normalized_col = (col - min_col) / (max_col - min_col)
        normalized_df[i] = normalized_col  # 标准化后的数据
    normalized_df['二分类'] = df['二分类']
    return normalized_df

normalized_df = normalize_features(df)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



best_svm = SVC(C=0.0001, kernel='rbf',decision_function_shape='ovo',gamma=np.float64(0.0001))
                       # kernel换成linear可以运行
best_svm.fit(X_train,y_train)
rfe = RFE(estimator=best_svm, n_features_to_select=10)  # 假设我们想保留10个特征
rfe = rfe.fit(X_train, y_train)
print("保留的特征索引:", rfe.support_)
print("保留的特征:", X_train.columns[rfe.support_])
X_train_rfe = X_train[:, rfe.support_]
X_test_rfe = X_test[:, rfe.support_]

svc.fit(X_train_rfe, y_train)
y_pred = best_svm.predict(X_test_rfe)

print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

plt.barh(range(len(rfe.ranking_)), rfe.ranking_)
plt.yticks(range(len(X_train.columns)), X_train.columns)
plt.show()