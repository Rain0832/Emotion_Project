import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"Data.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)

columns_to_drop = ['二分类']
X = df.iloc[:, :50]  # (892, 49)     df.iloc[:,0:50]第1~50列切片
y = df[columns_to_drop]
y = y.values.flatten()  # 一维列表

# 划分训练集与测试集
X_train, y_train = X[:600], y[:600]  # 前600行数据(0~599)是训练集，600份，X_train样本，y_train是标签
X_test, y_test = X[600:], y[600:]  # 600行以及之后的(600~891)是测试集，292份
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
svm_clf = Pipeline([
    ('first', StandardScaler()),  # first第一步，数据标准化
    ('second', SVC(kernel='rbf', C=1, random_state=64, gamma=0.1))  # second是第二步名称，kernel='rbf'高斯径向基核函数
])
"""

"""
# 训练模型
svm_clf.fit(X_train, y_train)  # 输入训练集以及对应标签

# 取第713号的数据，作为新样本，预测结果
predict_value = svm_clf.predict([X.iloc[713].values]) # [X.iloc[713].values]二维数组
# print("predict: ",predict_value)
# predict_value = 0
if predict_value==1:
    conclusion = "积极！"
    print("积极!")
else:
    conclusion = "消极！"
    print("消极!")

