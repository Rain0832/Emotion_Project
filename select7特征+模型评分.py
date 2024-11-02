"""决策树
每扔一个特征，重新训练一个模型，再扔下一个特征，直到数目够了
超参数不是不变的，每一轮超参数都是新的
保留几个特征？
"""
import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
columns_to_drop = ['二分类']
y = df[columns_to_drop]
y = y.values.flatten()  # 一维列表
X = df.iloc[:, :50]  # (892, 49)
X_train, y_train = X[:600], y[:600]  # 前600行数据(0~599)
X_test, y_test = X[600:], y[600:]  # 600行以及之后的(600~891)


# 创建参数空间
tree_param = {'ccp_alpha': np.arange(0, 1, 0.1).tolist(),
              'max_depth': np.arange(2, 8, 1).tolist(),
              'min_samples_split': np.arange(2, 5, 1).tolist(),
              'min_samples_leaf': np.arange(1, 4, 1).tolist(),
              'max_leaf_nodes': np.arange(10, 20, 1).tolist()}

# 实例化网格搜索评估器
tree_model = DecisionTreeClassifier()
tree_search_RFE = GridSearchCV(estimator=tree_model,
                               param_grid=tree_param,
                               n_jobs=12)

# 训练网格搜索评估器
tree_search_RFE.fit(X_train, y_train)

# 创建容器
rfe_rs1_cols = []
rfe_rs1_train = []
rfe_rs1_test = []

# 执行循环
for i in tqdm(range(49)):#0~49 ,应为 特征-1

    i = 49 - i

    # 首次循环时，创建X_train_temp和X_test_temp
    if i == 49:
        X_train_temp = (X_train).copy()
        X_test_temp = (X_test).copy()

        # 执行RFE过程，并计算不同特征子集下训练集和测试集的模型评分
    rfe_search = RFE(estimator=tree_search_RFE.best_estimator_, n_features_to_select=i).fit(X_train_temp, y_train)
    rfe_rs1_train.append(rfe_search.score(X_train_temp, y_train))
    rfe_rs1_test.append(rfe_search.score(X_test_temp, y_test))

    # 修改特征子集
    X_train_temp = X_train[rfe_search.get_feature_names_out()]
    X_test_temp = X_test[rfe_search.get_feature_names_out()]

    # 记录本轮循环被剔除的特征
    rfe_rs1_cols.append(rfe_search.feature_names_in_[rfe_search.ranking_ != 1])

print("rfe_rs1_train",rfe_rs1_train)#rfe_rs1_train [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
print("rfe_rs1_test",rfe_rs1_test)# rfe_rs1_test [0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384, 0.5616438356164384]



'''
# 创建参数空间
tree_param = {'ccp_alpha': np.arange(0, 1, 0.1).tolist(),
              'max_depth': np.arange(2, 8, 1).tolist(), 
              'min_samples_split': np.arange(2, 5, 1).tolist(), 
              'min_samples_leaf': np.arange(1, 4, 1).tolist(), 
              'max_leaf_nodes':np.arange(10,20, 1).tolist()}

# 创建容器
rfe_rs2_cols = []
rfe_rs2_train = []
rfe_rs2_test = []

# 执行循环
for i in tqdm(range(49)):
    
    i = 49 - i
    
    # 实例化模型评估器
    tree_model = DecisionTreeClassifier()
    tree_search_RFE = GridSearchCV(estimator = tree_model,
                                   param_grid = tree_param,
                                   n_jobs = 12)
    
    # 首次循环时，创建X_train_temp和X_test_temp
    if i == 49:
        X_train_temp = (X_train).copy()   
        X_test_temp = (X_test).copy()   
    
    # 训练模型
    tree_search_RFE.fit(X_train_temp, y_train)
    # 带入RFE过程
    rfe_search = RFE(estimator=tree_search_RFE.best_estimator_, n_features_to_select=i).fit(X_train_temp, y_train)
    # 计算当前子集情况下训练集和测试集评分
    rfe_rs2_train.append(rfe_search.score(X_train_temp, y_train))
    rfe_rs2_test.append(rfe_search.score(X_test_temp, y_test))
    # 修改数据集
    X_train_temp = X_train[rfe_search.get_feature_names_out()]   
    X_test_temp = X_test[rfe_search.get_feature_names_out()] 
        
    # 记录本轮循环被剔除的特征
    rfe_rs2_cols.append(rfe_search.feature_names_in_[rfe_search.ranking_ != 1])
print("rfe_rs1_train",rfe_rs1_train)
print("rfe_rs1_test",rfe_rs1_test)
'''