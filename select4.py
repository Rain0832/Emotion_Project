"""''''''''
决策树
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

# 实例化决策树评估器
tree_model = DecisionTreeClassifier()
# # 构造参数空间
# tree_param = {'ccp_alpha': np.arange(0, 1, 0.1).tolist(),
#               'max_depth': np.arange(2, 8, 1).tolist(),
#               'min_samples_split': np.arange(2, 5, 1).tolist(),
#               'min_samples_leaf': np.arange(1, 4, 1).tolist(),
#               'max_leaf_nodes':np.arange(10,20, 1).tolist()}
# # 实例化网格搜索评估器
# tree_search_RFE = GridSearchCV(estimator = tree_model,
#                                param_grid = tree_param,
#                                n_jobs = 12)#先训练一个不过拟合的、经超参数优化的模型

# tree_search_RFE.fit(X_train, y_train)
# print("best_params",
#       tree_search_RFE.best_params_)  #best_params {'ccp_alpha': 0.1, 'max_depth': 2, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
# print("best_estimator_",
#       tree_search_RFE.best_estimator_)  #best_estimator_ DecisionTreeClassifier(ccp_alpha=0.1, max_depth=2, max_leaf_nodes=10)
tree_param = {'ccp_alpha': [0.1],
              'max_depth': [2],
              'min_samples_split': [2],
              'min_samples_leaf':[1] ,
              'max_leaf_nodes': [10]}
tree_search_RFE = GridSearchCV(estimator=tree_model,
                               param_grid=tree_param,
                               n_jobs=12)

tree_search_RFE.fit(X_train, y_train)
rfe_search = RFE(estimator=tree_search_RFE.best_estimator_,n_features_to_select=1).fit(X_train,y_train)
print("rank:",rfe_search.ranking_)
