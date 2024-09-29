from scipy.fftpack import fft  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def moving_average(signal, window_size):
        # 使用卷积计算移动平均
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
def extract_gsr_features_former(signal):
    # 1. 计算信号的一阶导数
    derivative = np.diff(signal)  
    # 2. 计算导数的平均值
    average_derivative = np.mean(derivative)
    # 3. 计算导数中负样本的百分比
    negative_samples = derivative[derivative < 0]  # 提取负样本
    negative_percent_in_deriv = len(negative_samples) / len(derivative) * 100  # 计算百分比
    # 4. 查找局部极小值的数量

    # 应用移动平均，窗口大小为20
    smoothed_signal = moving_average(signal, 20)

    # 首先将 signal 转换为 NumPy 数组
    signal_np = np.array(smoothed_signal)
    local_minima_indices = argrelextrema(signal_np, np.less,order=10)[0]  # 查找局部极小值的索引# order:两侧使用多少点进行比较
    number_of_local_minima = len(local_minima_indices)  # 计算数量

    diff_signal = np.diff(signal)  # 计算一阶差分信号
    diff_std_dev = np.std(diff_signal)  # 计算一阶差分标准差
    return (average_derivative,negative_percent_in_deriv,number_of_local_minima,diff_std_dev)


# 提取PPG特征：1
def extract_ppg_features(signal_normalized):
    ppg = signal_normalized #归一化的PPG
    min_distance = 65  # 两个峰最小间距
    min_height = 0.2 # 峰值最小高度
    min_low = 0.85
    peaks, _ = find_peaks(ppg, distance=min_distance, height=min_height)
    troughs, _ = find_peaks(-ppg, distance=min_distance, height=-min_low)
    if peaks[0] < troughs[0]:
        peaks_7 = peaks[1:8]
        troughs_7 = troughs[0:7]
    else:
        peaks_7 = peaks[0:7]
        troughs_7 = troughs[0:7]
    # 从信号中获取峰值的高度
    peak_heights = ppg[peaks_7]
    troughs_height = ppg[troughs_7]
    high = peak_heights - troughs_height
    aver_high = np.mean(high)

    return aver_high


# GSR特征写入excel
def gsr_features_to_excel():
    gsr_features = []
    gsr_total = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\a_preprocessed_GSR_final.xlsx",
                              index_col=0,header=0)
    for i in range(0, 1):  # 一行一行提取特征
        gsr = gsr_total.iloc[i, :]  # 第i行所有列
        np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        average_derivative,negative_percent_in_deriv,local_minima_number,diff_std_dev = extract_gsr_features_former(np_gsr)  # 频域特征的第一行:1x3

        single_line = (average_derivative,negative_percent_in_deriv,local_minima_number,diff_std_dev)
        gsr_features.append(single_line)
    resultPath1 = r"D:\桌面\Gitfile\NewWeb\Data\b_GSRFeature.xlsx"  # 指定excel的路径,
    df2 = pd.DataFrame(gsr_features)  # 将gsr_featurese变成DataFrame形式
    df2.to_excel(resultPath1, sheet_name="GSR4个特征（带行列索引）")


    # 返回所有特征的列表
    return 1


# PPG特征写入excel
def ppg_features_to_excel():
    ppg_features = []
    ppg_total = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\NormPPG.xlsx",index_col=0,header=0)
    
    # DataFrame形式
    for i in range(0, 1):  # 一行一行提取特征
        ppg = ppg_total.iloc[i, :]  # 第i行所有列
        np_ppg = np.array(ppg).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        aver_high = extract_ppg_features(np_ppg)  # 频域特征的第一行:1x3
        single_line = (aver_high)
        ppg_features.append(single_line)
    resultPath3 = r"D:\桌面\Gitfile\NewWeb\Data\b_PPGFeature.xlsx"  # 指定excel的路径,
    df3 = pd.DataFrame(ppg_features)  # 将ppg_featurese变成DataFrame形式
    df3.to_excel(resultPath3, sheet_name="PPG1个特征（带行列索引）")
    return 1

# 准备特征数据集
# def prepare_feature_dataset(all_data, all_labels):
if __name__ == '__main__':
    a,b = 0,0
    a = gsr_features_to_excel()
    b = ppg_features_to_excel()
    if a == 1:
        print("GSR特征成功写入excel！")
    else:
        print("GSR特征写入excel失败")
    if b == 1:
        print("PPG特征成功写入excel！")
    else:
        print("PPG特征写入excel失败")



