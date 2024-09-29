import pandas as pd

# 读取Excel文件
df = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\rawPPG.xlsx", index_col=0, header=0)

normalized = []

# 对每一列数据进行标准化
normalized_df = pd.DataFrame()
for col in df.columns:  # 逐列处理
    single_column = df[col]
    min_col = single_column.min()
    max_col = single_column.max()
    # 归一化公式 (x - min) / (max - min)
    new_column = (single_column - min_col) / (max_col - min_col)
    normalized.append(new_column)

# 将归一化后的列转为DataFrame
normalized_df = pd.DataFrame(normalized).T  # 需要转置为列格式

# 指定保存路径
resultPath1 = r"D:\桌面\Gitfile\NewWeb\Data\NormPPG.xlsx"
normalized_df.to_excel(resultPath1, sheet_name="归一化数据", index=True)
print("PPG归一化成功")
