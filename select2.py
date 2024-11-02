'''
计算F值以及概率 +进行方差分析筛选k个特征（用离散型标签筛选连续型特征，根据最小的p或最大的F_score）
只训练一个模型，根据特征的重要程度扔特征
'''
import numpy as np
import pandas as pd
import scipy.special
from sklearn.svm import SVC

df = pd.read_excel(r"E:\2大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
columns_to_drop = ['二分类']
y = df[columns_to_drop]
y = y.values.flatten()  # 一维列表
# 特征标准化
normalized_df = pd.DataFrame()
for i in range(0, 50):  #一列一列标准化
    col = df.iloc[:, i]
    min_col = col.min()
    max_col = col.max()
    normalized_col = (col - min_col) / (max_col - min_col)
    normalized_df[i] = normalized_col  # 标准化后的数据
normalized_df['二分类'] = df['二分类']
#print(normalized_df[0])

# resultPath1 = r"E:\大学项目\选择合适的特征\标准化数据_最小最大值（带行列索引）.xlsx"  # 指定excel的路径,
# normalized_df.to_excel(resultPath1, sheet_name="标准化数据（带行列索引）")

#按标签将特征1分为两类
cat1, cat0 = [], []
for index, single_line in normalized_df.iterrows():  # .iterrows() 返回的是一个包含两个元素的元组：索引和行数据。
    feature1 = single_line[0]
    if single_line['二分类'] == 0:
        cat0.append(feature1)
    else:
        cat1.append(feature1)  # 至此，cat1是标签为1对应的数据
feature1 = normalized_df[0]  # Length: 892,
feature1to50 = normalized_df.iloc[:, 0:50]

'''
F检验：                                               情绪：    0    1
  F检验=方差齐性检验= ANOVA方差分析前提：            此特征数值：    ..   ..     
1.各样本是相互独立的随机样本，均服从正态分布;                         ..  ..
2.各样本的总体方差相等，即方差齐性。                                 ..  ..        
H0：情绪的正性与负性与这一个特征没有关系                             ..   ..
若cat0与cat1来自同一样本空间，则F的值在1附近，
分子自由度：分子上的样本容量-1
分母自由度：分母上的样本容量-1
alpha=0.1 ?
'''

def compute_F_score_and_p(cat0, cat1):
    cat0_mean = np.mean(cat0)
    SSE0 = np.power(cat0 - cat0_mean, 2).sum()  # SSW 组内离差平方和
    cat1_mean = np.mean(cat1)
    SSE1 = np.power(cat1 - cat1_mean, 2).sum()
    SSE = SSE1 + SSE0

    n0, n1 = len(cat0), len(cat1)
    cat_mean = np.mean(feature1)
    SSB = n0 * np.power(cat0_mean - cat_mean, 2) + n1 * np.power(cat1_mean - cat_mean, 2)
    # SSB 组间离差平方和
    SST = np.power(feature1 - cat_mean, 2).sum()  # SSE+SSB==SST 算总的离差平方和
    k, n = 2, len(feature1)  # k是标签的类别
    MSB = SSB / (k - 1)  # k-1，n-k是自由度
    MSE = SSE / (n - k)
    F_score = MSB / MSE  # F值越大，说明此特征与标签越有关，此时组内间隔小，组与组的间隔大
    print('F_score=', F_score)
    p = scipy.special.fdtrc(k - 1, n - k, F_score)  # 计算概率p
    print('p=', p)


# print(scipy.stats.f_oneway(cat1, cat0))  # 第一个特征的 F_score,概率，需输入标签为0，1的两类特征
from sklearn.feature_selection import f_classif, SelectKBest  #, SelectName

# print(f_classif(feature1.values.reshape(-1, 1), y))  #只需输入某一特征+标签，不必按标签分类 .reshape(-1,1)注意是二维的
#噪音检验
# np.random.seed(56)
# noisy0 = np.random.rand(892)
# np.random.seed(45)
# noisy1= np.random.rand(892)
# print(scipy.stats.f_oneway(noisy1,noisy0))  # F_score,概率

# 筛选特征

KB_CF = SelectKBest(f_classif, k=2)  # k=2，选取两个最重要特征
KB_CF.fit(feature1to50, y)  # SelectKBest(k=2)
selected_features = KB_CF.get_support()  # 对每一个特征，展示true 还是false
print("the Trues are:\n", selected_features)

print("KB_CF.scores_\n", KB_CF.scores_)
print("KB_CF.pvalues_\n", KB_CF.pvalues_)
# SelectName(KB_CF)
# SelectName_P(P=0.01, KB=KB_CF)

