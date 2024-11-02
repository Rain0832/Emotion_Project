"""SVM
每扔一个特征，重新训练一个模型，再扔下一个特征，直到数目够了
先训练一个不过拟合的、经超参数优化的模型，超参数是继承的，一个模型用到底
"""
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

# 定义SVM模型
svm_model = SVC(decision_function_shape='ovo')

# # 构造参数网格
# param_grid = {
#     'C': np.arange(0.0001, 0.01, 0.001),  # 正则化参数
#     'kernel': ['rbf'],  # 核函数类型
#     'gamma': np.arange(0.0001,0.01, 0.001)  # 核函数的系数
# }
# # 实例化网格搜索评估器
# grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
#                            #cv 交叉验证的折数，分为几组，n_jobs=-1：使用所有可用的CPU核心。verbose=2：控制输出的详细程度，2表示更详细的输出。
# grid_search_svm.fit(X_train, y_train)# 在训练集上，执行网格搜索
# # 打印最佳参数和最佳分数
# print("Best parameters found:", grid_search_svm.best_params_) #Best parameters found: {'C': np.float64(0.1), 'gamma': 'scale', 'kernel': 'rbf'}
# print("Best cross-validation score:", grid_search_svm.best_score_)#Best cross-validation score: 0.6

# # 获取最佳SVM模型
# best_svm = grid_search_svm.best_estimator_
# print("best svm is", best_svm) #best svm is SVC(C=np.float64(0.1), decision_function_shape='ovo')

best_svm = SVC(C=0.0001, kernel='rbf',decision_function_shape='ovo',gamma=np.float64(0.0001))
# 实例化RFE，使用最佳SVM模型作为估计器
rfe_search = RFE(estimator=best_svm, n_features_to_select=1)
# 训练RFE
rfe_search.fit(X_train, y_train)

# 打印特征排名
print("Feature ranking:", rfe_search.ranking_)
