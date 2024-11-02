import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import datetime

df = pd.read_excel(r"E:\大学项目\二分类特征+标签\筛选后特征+标签（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows= 8064)

#print("df",df)
columns_to_drop = ['二分类']
# print(df.columns)
X = df.iloc[:,0:50]#地1~50列切片
y = df[columns_to_drop]
y = y.values.ravel()
#print("x",X.iloc[:,1])
print(np.unique(y))
#plt.scatter(X.iloc[:,1],X.iloc[:,4],c=y)   # X[:,0] 和 X[:,1]，这实际上是在尝试访问名为 "0" 和 "1" 的列，而不是按位置索引。
#plt.show()                                    #

#主成分分析：
from  sklearn.datasets import load_iris
from sklearn.decomposition import PCA
X_dr=PCA(2).fit_transform(X)
# from sklearn.preprocessing import LabelEncoder
#
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
#
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
#
#
# from sklearn.svm import SVC
# svc_model = SVC(kernel = 'linear')
# comptd = svc_model.fit(X_train, y_train)
# #accuracy = comptd.score(X_train, y_train)
#
# accuracy = comptd.score(X_test, y_test)
# print('accuracy=',accuracy)
# # print(svc_model)
# # print(svc_model.kernel_params_)
#
