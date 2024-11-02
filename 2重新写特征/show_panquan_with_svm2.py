import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)


X = df.iloc[:, :-1].values  # 特征 (892, 50)
y = df.iloc[:, -1].values  # 目标 (892,1)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


"""""
当然也可以这样划分训练集与测试集：

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)

train_test_split()函数: 用于将数据集划分为训练集train和测试集test
X: 待划分的样本特征集
y: 数据集X对应的标签
test_size: 0~1表示测试集样本占比、整数表示测试集样本数量
random_state: 随机数种子。在需要重复实验的时候保证得到一组一样的随机数据。每次填1(其他参数一样)，每次得到的随机数组一样；每次填0/不填，每次都不一样
stratify=y: 划分数据集时保证每个类别在训练集和测试集中的比例与原数据集中的比例相同
"""


# 创建一个管道，构建模型
svm_clf = SVC(C=9.9, kernel='rbf', gamma=10.5, decision_function_shape='ovo')
"""

"""
# 训练模型
svm_clf.fit(X_train, y_train)  # 输入训练集以及对应标签

# 取第713号的数据，作为新样本，预测结果
predict_value = svm_clf.predict([X[413]]) # 二维数组
if predict_value==1:
    print("积极!")
else:
    print("消极!")

