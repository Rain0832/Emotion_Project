"""决策树
每扔一个特征，重新训练一个模型，再扔下一个特征，直到数目够了
超参数不是不变的，每一轮超参数都是新的
50min
"""
import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

df = pd.read_excel(r"E:\2大学项目\选择合适的特征\seek_for_features\筛选后特征+标签2（列索引有名称）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
columns_to_drop = ['二分类']
y = df[columns_to_drop]
y = y.values.flatten()  # 一维列表
X = df.iloc[:, :50]  # (892, 49)
X_train, y_train = X[:600], y[:600]  # 前600行数据(0~599)
X_test, y_test = X[600:], y[600:]  # 600行以及之后的(600~891)
# 定义参数空间
tree_param = {'ccp_alpha': np.arange(0, 1, 0.1).tolist(),
              'max_depth': np.arange(2, 8, 1).tolist(),
              'min_samples_split': np.arange(2, 5, 1).tolist(),
              'min_samples_leaf': np.arange(1, 4, 1).tolist(),
              'max_leaf_nodes': np.arange(10, 20, 1).tolist()}

# 创建容器
rfe_res_search1 = []

# 执行循环
for i in tqdm(range(49)):  #显示进度条

    i = 49 - i
    # 实例化网格搜索评估器
    tree_model = DecisionTreeClassifier()
    tree_search_RFE = GridSearchCV(estimator=tree_model,
                                   param_grid=tree_param,
                                   n_jobs=12)

    # 首次循环时，创建X_train_temp，包含50个特征
    if i == 49:
        X_train_temp = X_train.copy()

        # 训练模型，然后带入RFE评估器
    tree_search_RFE.fit(X_train_temp, y_train)
    rfe_search = RFE(estimator=tree_search_RFE.best_estimator_, n_features_to_select=i).fit(X_train_temp, y_train)
    X_train_temp = X_train[rfe_search.get_feature_names_out()]   # 更新训练数据集，只保留选中的特征。而扔掉剔除的特征
    # 搜索本轮被淘汰的特征，并记入rfe_res_search1
    rfe_res_search1.append(rfe_search.feature_names_in_[rfe_search.ranking_ != 1])

# 清除临时变量
gc.collect()
print(rfe_search.get_feature_names_out)
print("先后剔除：",rfe_res_search1)   # 上面的是先扔掉的

rfe_res_search = rfe_search.get_feature_names_out().tolist() + np.array(rfe_res_search1[::-1]).flatten().tolist()
print("重要性降序：",rfe_res_search)                                                 # rfe_res_search1[::-1]颠倒排序

# #r"""
# E:\python312\python.exe E:\PYTHON代码\daily_learn_SVM\select6超参数++.py
#   0%|          | 0/49 [01:21<?, ?it/s]
# Traceback (most recent call last):
#   File "E:\PYTHON代码\daily_learn_SVM\select6超参数++.py", line 45, in <module>
#     X_train_temp = X_train[rfe_search.get_feature_names_out()]
#                    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\python312\Lib\site-packages\pandas\core\frame.py", line 4108, in __getitem__
#     indexer = self.columns._get_indexer_strict(key, "columns")[1]
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\python312\Lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
#     self._raise_if_missing(keyarr, indexer, axis_name)
#   File "E:\python312\Lib\site-packages\pandas\core\indexes\base.py", line 6249, in _raise_if_missing
#     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
# KeyError: "None of [Index(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',\n       'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',\n       'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31',\n       'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41',\n       'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49'],\n      dtype='object')] are in the [columns]"
#
# 进程已结束，退出代码为 1

# """

# """
# E:\python312\python.exe E:\PYTHON代码\daily_learn_SVM\select6超参数++.py
# E:\PYTHON代码\daily_learn_SVM\select6超参数++.py:56: SyntaxWarning: invalid escape sequence '\p'
#   '''
#  98%|█████████▊| 49/50 [40:35<00:49, 49.71s/it]
# Traceback (most recent call last):
#   File "E:\PYTHON代码\daily_learn_SVM\select6超参数++.py", line 44, in <module>
#     X_train_temp = (X_train).copy()
#              ^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\python312\Lib\site-packages\sklearn\base.py", line 1466, in wrapper
#     estimator._validate_params()
#   File "E:\python312\Lib\site-packages\sklearn\base.py", line 666, in _validate_params
#     validate_parameter_constraints(
#   File "E:\python312\Lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
#     raise InvalidParameterError(
# sklearn.utils._param_validation.InvalidParameterError: The 'n_features_to_select' parameter of RFE must be None, a float in the range (0.0, 1.0] or an int in the range (0, inf). Got 0 instead.
#
# 进程已结束，退出代码为 1
#
# """