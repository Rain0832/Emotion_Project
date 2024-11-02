import pandas as pd
import numpy as np

previous_data = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+5特征.xlsx",
                              index_col=0, nrows=None)
selected_data = []
abandoned = 0  # 舍弃的样本数
for index, single_line in previous_data.iterrows():  # 将愉悦度分为两类：6-9为正性（1），1-4为负性（0）
    df_label = single_line["二分类"]
    if df_label > 6:
        pleasure = 1
    elif df_label < 4:
        pleasure = 0
    else:
        pleasure = -100
        abandoned += 1
        continue
    single_line["二分类"] = pleasure
    if pleasure != -100:  # 若pleasure==100，则single_line虽被改值为100，但不会被写入
        selected_data.append(single_line)

print("应该舍弃", abandoned, "个样本")
resultPath2 = r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\879筛选后4特征+5特征+标签.xlsx"  # 指定excel的路径,
df2 = pd.DataFrame(selected_data)
df2.reset_index(drop=True, inplace=True)  # 重置行索引，丢弃旧的索引
df2.to_excel(resultPath2, sheet_name="879筛选后特征+标签（带行列索引）")

