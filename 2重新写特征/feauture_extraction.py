from scipy.fftpack import fft  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd
from scipy.signal import argrelextrema
# 提取GSR特征：3个
#
def moving_average(signal, window_size):
        # 使用卷积计算移动平均
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
def extract_gsr_features_former(signal):
    # 1. 计算导数
    derivative = np.diff(signal)  # 计算信号的一阶导数
    # 2. 计算导数的平均值
    average_derivative = np.mean(derivative)
    # 3. 计算导数中负样本的百分比
    negative_samples = derivative[derivative < 0]  # 提取负样本
    negative_percent_in_deriv = len(negative_samples) / len(derivative) * 100  # 计算百分比
    # 4. 查找局部极小值的数量

    # 应用移动平均，窗口大小为12
    smoothed_signal = moving_average(signal, 20)

    # 首先将 signal 转换为 NumPy 数组
    signal_np = np.array(smoothed_signal)
    local_minima_indices = argrelextrema(signal_np, np.less,order=10)[0]  # 查找局部极小值的索引# order:两侧使用多少点进行比较
    number_of_local_minima = len(local_minima_indices)  # 计算数量
    return (average_derivative,negative_percent_in_deriv,number_of_local_minima)

def extract_gsr_features_latter(signal):
    # 1. 计算导数
    derivative = np.diff(signal)  # 计算信号的一阶导数
    # 2. 计算导数的平均值
    average_derivative = np.mean(derivative)
    # 3. 计算导数中负样本的百分比
    negative_samples = derivative[derivative < 0]  # 提取负样本
    negative_percent_in_deriv = len(negative_samples) / len(derivative) * 100  # 计算百分比
    # 4. 查找局部极小值的数量

    # 应用移动平均，窗口大小为12
    smoothed_signal = moving_average(signal, 700)

    # 首先将 signal 转换为 NumPy 数组
    signal_np = np.array(smoothed_signal)
    local_minima_indices = argrelextrema(signal_np, np.less,order=400)[0]  # 查找局部极小值的索引# order:两侧使用多少点进行比较
    number_of_local_minima = len(local_minima_indices)  # 计算数量
    return (average_derivative,negative_percent_in_deriv,number_of_local_minima)

# 提取PPG特征：2个
def extract_ppg_features(signal):

    return

## 二分类标签
def pleasure_label_to_excel():
    pleasure_labels = []
    df_labels = pd.read_excel(r"E:\2大学项目\特征提取——数据与程序\所有的标签.xlsx", usecols=["唤醒度"])
    # index_col = 0不要行索引,只读唤醒度列
    # 最好保留标签
    df_labels = df_labels.reset_index(drop=True)  # 不要原索引，并重置索引，1280x1
    # print(df_labels)
    # 二分类标签存excel文件
    df_labels["唤醒度"] = pd.to_numeric(df_labels["唤醒度"], errors='coerce')
    # 转成数字形式，errors='coerce': 当转换过程中遇到无法转换为数值的数据时,Pandas将这些数据转换为 NaN（不是数字）
    abandoned = 0  # 舍弃的样本数
    for df_label in df_labels["唤醒度"]:  # 将愉悦度分为两类：6-9为正性（1），1-4为负性（0）
        if df_label > 6:
            pleasure = 1
        elif df_label < 4:
            pleasure = 0
        else:
            pleasure = -100
            abandoned = abandoned + 1

        pleasure_labels.append(df_label)
    print("应该舍弃", abandoned, "个样本")
    resultPath1 = r"E:\2大学项目\2重新写特征\二分类原始标签（带索引）.xlsx"  # 将二分类标签导出excel
    column_names = ["二分类"]  # 列名称为二分类
    df1 = pd.DataFrame(pleasure_labels, columns=column_names)
    df1.to_excel(resultPath1, sheet_name="二分类原始标签（带索引）")
    return 555
# GSR特征写入excel
def gsr_features_to_excel():
    ##gsr
    gsr_features = []
    gsr_total = pd.read_excel(r"E:\2大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx",
                              index_col=0, usecols=range(0, 8065), nrows=None)
    ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    ##DataFrame形式
    for i in range(0, 880):  # 一行一行提取特征
        gsr = gsr_total.iloc[i, :]  # 第i行所有列
        np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        average_derivative,negative_percent_in_deriv,local_minima_number = extract_gsr_features_former(np_gsr)  # 频域特征的第一行:1x3

        single_line = (average_derivative,negative_percent_in_deriv,local_minima_number)
        gsr_features.append(single_line)
    for i in range(880, 1280):
        gsr = gsr_total.iloc[i, :]  # 第i行所有列
        np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        average_derivative, negative_percent_in_deriv, local_minima_number = extract_gsr_features_latter(
            np_gsr)  # 频域特征的第一行:1x3
        single_line = (average_derivative, negative_percent_in_deriv, local_minima_number)
        gsr_features.append(single_line)
    resultPath1 = r"E:\2大学项目\2重新写特征\GSR3个特征（带行列索引）.xlsx"  # 指定excel的路径,
    df2 = pd.DataFrame(gsr_features)  # 将gsr_featurese变成DataFrame形式
    df2.to_excel(resultPath1, sheet_name="GSR3个特征（带行列索引）")


    # 返回所有特征的列表
    return 666
#ppg特征写入excel
def ppg_features_to_excel():
    ##ppg
    ppg_features = []
    # ppg_total = pd.read_excel(r"E:\大学项目\所有数据excel\所有原PPG数据（带行列索引）.xlsx",
    #                           index_col=0, usecols=range(0, 8065), nrows=None)
    # ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
    # ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    # ##DataFrame形式
    # for i in range(0, 1280):  # 一行一行提取特征
    #     ppg = ppg_total.iloc[i, :]  # 第i行所有列
    #     np_ppg = np.array(ppg).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
    #     # 将DataFrame形式转换成numpy数组形式
    #     peak_freq, mean_freq, avg_power_freq = ppg_features_to_excel(np_ppg)  # 频域特征的第一行:1x3
    #     single_line = ()
    #     ppg_features.append(single_line)
    # resultPath3 = r"E:\大学项目\特征提取\PPG所有特征（带行列索引）.xlsx"  # 指定excel的路径,
    # df3 = pd.DataFrame(ppg_features)  # 将ppg_featurese变成DataFrame形式
    # df3.to_excel(resultPath3, sheet_name="PPG所有特征（带行列索引）")
    return 777

# 准备特征数据集
#def prepare_feature_dataset(all_data, all_labels):
if __name__ == '__main__':
    #先pip install xlrd,pip install openpyxl
    a,b,label = 0,0,0
    # label = pleasure_label_to_excel()
    a = gsr_features_to_excel()
    # b = ppg_features_to_excel()
    if label == 555:
        print("二分类标签成功写入excel！")
    else:
        print("二分类标签写入excel失败")
    if a == 666:
        print("gsr特征成功写入excel！")
    else:
        print("gsr特征写入excel失败")
    if b == 777:
        print("ppg特征成功写入excel！")
    else:
        print("ppg特征写入excel失败")



