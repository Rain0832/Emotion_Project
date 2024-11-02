import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"E:\大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
# 特征标准化
normalized_df = pd.DataFrame()
for i in range(0,50):
    col = df.iloc[:, i]
    min_col = col.min()
    max_col = col.max()
    normalized_col = (col - min_col) / (max_col - min_col)
    normalized_df[i] = normalized_col # 标准化后的数据

np.random.seed(56)
noisy = np.random.rand(892)
noisy_mean = np.mean(noisy)
print("Mean of noisy array:", noisy_mean)
noisy_variance = np.var(noisy)# 方差
print("Variance of noisy array:", noisy_variance)

for j in range(0,50):
    column = normalized_df.iloc[:, j]
    column_mean =np.mean(column)
    column_variance = np.var(column)
    # if column_variance <= noisy_variance:
    #     print(j+1)
    #print(column_variance)
