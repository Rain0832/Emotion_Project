import pandas as pd
import numpy as np


# 读取数据
df_raw = pd.read_csv(r"E:\2大学项目\final_project_for_memory\sensor_data.txt", sep=',', header=0, usecols=[1,2])
# 假设我们关注的是第一列数据
column_data = df_raw.iloc[:, 1]
# 获取非NaN值的索引和数据
non_nan_indices = column_data.dropna().index
# bound = max(non_nan_indices)
length = len(non_nan_indices)
# non_nan_values = column_data.dropna().values
# print(non_nan_indices)

# 使用前50个数据不断循环代替NaN
def fill_nan_with_cycle(data, cycle_length=length ):
    # 获取非NaN值的索引和数据
    non_nan_indices = np.where(~np.isnan(data))[0]
    non_nan_values = data[non_nan_indices]

    # 如果非NaN值的数量小于循环长度，则使用所有非NaN值
    if len(non_nan_values) < cycle_length:
        cycle_length = len(non_nan_values)
    # 创建一个循环数组
    cycle_values = np.tile(non_nan_values[:cycle_length], int(np.ceil(len(data) / cycle_length)))
    # 填充NaN值
    filled_data = data.copy()
    nan_indices = np.isnan(data)
    filled_data[nan_indices] = cycle_values[:len(data)][nan_indices]
    return filled_data

# #使用循环数组填充NaN值
df_raw.iloc[:, 1] = fill_nan_with_cycle(column_data)


# 转置DataFrame
df_line = df_raw.transpose()

# 访问转置后的DataFrame的行
gsr = df_line.iloc[0]
ppg = df_line.iloc[1]
# print(gsr)
# 将Series转换为单列的DataFrame
gsr_df = pd.DataFrame(gsr).transpose()
ppg_df = pd.DataFrame(ppg).transpose()

# 写入Excel文件
resultPath1 = r"E:\2大学项目\final_project_for_memory\0.GSR数据转excel.xlsx"
gsr_df.to_excel(resultPath1, sheet_name="GSR数据转excel（带行列索引）", index=True)  # index=True: 保存索引列。

resultPath2 = r"E:\2大学项目\final_project_for_memory\0.PPG数据转excel.xlsx"
ppg_df.to_excel(resultPath2, sheet_name="PPG数据转excel（带行列索引）", index=True)  # index=True: 保存索引列。
print("txt转excel成功！")