"""''''''''决策树
RFE: 特征递归消除法
每扔一个特征，重新训练一个模型，再扔下一个特征，直到数目够了
但是每一轮的模型可能都是过拟合的！
"""
import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None,header=0) # 默认会将 Excel 文件的第一行作为列名
X = df.iloc[:, :-1].values  # 特征 (892, 50)     iloc: 基于整数位置的索引。[:, :-1]: 所有行，除最后一列外的所有列。
y = df.iloc[:, -1].values  # 目标变量 (892,)     所有行，最后一列。
X_train, y_train = X[:600], y[:600]  # 前600行数据(0~599)
X_test, y_test = X[600:], y[600:]  # 600行以及之后的(600~891)


# svm_model = SVC(C=10,kernel='rbf', gamma=10, decision_function_shape='ovo')

rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=1,importance_getter='auto')
rfe.fit(X_train, y_train)  # n_features_to_select=None 留下一半特征，importance_getter重要度
print("原始总特征数：", rfe.n_features_in_)
print("筛选后特征数：", rfe.n_features_)
print("rfe.get_support()", rfe.get_support())  # true为保留的
# 获取保留特征的原始列名
original_feature_names = df.columns[:-1]  # 去掉目标变量的列名
selected_feature_indices = rfe.get_support(indices=True)
selected_feature_names = original_feature_names[selected_feature_indices]
print("保留特征名称：", selected_feature_names.tolist())# 保留特征的名称

# print("保留特征名称：",rfe.get_feature_names_out())  # 保留特征的名称
print("rank:", rfe.ranking_) # 最大的表示第一轮就被扔了，最小的值1是保留的

# rfe.predict(X_train) # 将最后一轮保留的特征待入最后一轮训练的模型，预测y
print("训练集得分",rfe.score(X_train,y_train))
print("测试集得分",rfe.score(X_test,y_test))

# print( X_train[rfe.get_feature_names_out()[0]]      )