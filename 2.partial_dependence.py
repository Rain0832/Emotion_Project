import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.pyplot as plt
"""
partial dependence:部分依赖图
对某一特征，改变其值的大小，看预测结果的变化，趋势
"""

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

pdp_goals = pdp.pdp_isolate(model=svm_model, dataset=X_test, feature='特征1')
pdp.pdp_plot(pdp_goals, '特征1')
plt.show()
