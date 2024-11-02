import numpy as np
import pandas as pd
"""
恰好是0~1之间
"""
# 读取Excel文件
df = pd.read_excel(r"E:\2大学项目\lmh预处理\finally\preprocessed_PPG_final.xlsx",
                   index_col=0, usecols=range(0, 8065), nrows=None)
print(df)
normalized=[]

# 特征标准化
normalized_df = pd.DataFrame()
for i in range(1280):  # 一列一列标准化
    single_line = df.iloc[i,: ]
    min_si = single_line.min()
    max_si= single_line.max()
    new_line = (single_line-min_si)/(max_si-min_si)
    normalized.append(new_line)
# print(new_line)
normalized_df = pd.DataFrame(normalized)

# # 指定保存路径
resultPath1 = r"E:\2大学项目\2重新写特征\ppg标准化\ppg标准化数据_最小最大值（带行列索引）.xlsx"
normalized_df.to_excel(resultPath1, sheet_name="标准化数据（带行列索引）", index=True)
