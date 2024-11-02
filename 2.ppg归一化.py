import pandas as pd
"""
是0~1之间，归一化
"""
# 读取Excel文件
df = pd.read_excel(r"E:\2大学项目\final_project_for_memory\1.preprocessed_PPG_final.xlsx",
                   index_col=0, header=0)
# print(df)
normalized=[]

# 特征标准化
normalized_df = pd.DataFrame()
for i in range(1):  # 一列一列标准化
    single_line = df.iloc[i,: ]
    min_si = single_line.min()
    max_si= single_line.max()
    new_line = (single_line-min_si)/(max_si-min_si)
    normalized.append(new_line)
# print(new_line)
normalized_df = pd.DataFrame(normalized)

# # 指定保存路径
resultPath1 = r"E:\2大学项目\final_project_for_memory\2.ppg归一化数据_最小最大值（带行列索引）.xlsx"
normalized_df.to_excel(resultPath1, sheet_name="归一化数据（带行列索引）", index=True)
print("ppg已经归一化")