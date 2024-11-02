import numpy as np
import pandas as pd
"""
恰好是0~1之间
"""
# 读取Excel文件
df = pd.read_excel(r"E:\大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)

feature_names = ['特征' + str(i + 1) for i in range(50)]# 将列索引重命名

# 特征标准化
normalized_df = pd.DataFrame()
for i in range(50):  # 一列一列标准化
    col = df.iloc[:, i]
    min_col = col.min()
    max_col = col.max()
    normalized_df[feature_names[i]] = (col - min_col) / (max_col - min_col)  # 直接使用新列名

# 将'二分类'列添加到标准化后的DataFrame
normalized_df['二分类'] = df['二分类']

# 指定保存路径
resultPath1 = r"E:\2大学项目\预处理后数据的标准化\标准化数据_最小最大值（带行列索引）.xlsx"
normalized_df.to_excel(resultPath1, sheet_name="标准化数据（带行列索引）", index=True)
