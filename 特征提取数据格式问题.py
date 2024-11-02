from scipy.fftpack import fft  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd

def extract_frequency_domain_features(signal):
    fft_values = abs(fft(np_gsr))  # 取其模值 ，纵坐标 fft_values.shape (8064,)
    half_of_gsr = len(np_gsr) // 2  # =4032 整数除法
    peak_index = np.argmax(fft_values[1:half_of_gsr]) + 1  # 取一半
    freqs = np.fft.fftfreq(len(np_gsr), d=1.0 / 128)  # freqs (8064,) 计算对应横坐标:频率
    peak_freq = freqs[peak_index]  # 峰值频点

    mean_freq = np.mean(freqs)  # 计算平均频点 有问题
    avg_power_freq = np.sum(fft_values ** 2) / len(signal)  # 计算平均功率频率
    return peak_freq, mean_freq, avg_power_freq

# 提取GSR和PPG信号的特征
def extract_features(gsr_signal):

    gsr_time_features = extract_time_domain_features(gsr_signal)  # 提取GSR的时域特征
    gsr_frequency_features = extract_frequency_domain_features(gsr_signal)  # 提取GSR的频域特征
    #ppg_time_features = extract_time_domain_features(ppg_signal)  # 提取PPG的时域特征
    #ppg_frequency_features = extract_frequency_domain_features(ppg_signal)  # 提取PPG的频域特征

    # 返回所有特征的列表
    return gsr_time_features + gsr_frequency_features


if __name__ == '__main__':

    ##gsr
    gsr_features=[]
    gsr_total=pd.read_excel(r"E:\大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx",
                               index_col=0,usecols=range(0,8065),nrows=2)
       ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
       ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
       ##DataFrame形式
    for i in range(0,1280):   #一行一行提取特征
       gsr=gsr_total.iloc[i, :]
       np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
                                           #将DataFrame形式转换成numpy数组形式
       peak_freq, mean_freq, avg_power_freq= extract_frequency_domain_features(np_gsr) # 频域特征的第一行:1x3
       single_line=peak_freq, mean_freq, avg_power_freq
       gsr_features.append(single_line)
    print("single_line",single_line)
    print("gsr_features len：",len(gsr_features))
    print("gsr_features",gsr_features)

    # resultPath1 = r"E:\大学项目\特征提取\gsr频域特征.xlsx"  # 指定excel的路径,
    # df1 = pd.DataFrame(gsr_features)  # 将gsr_featurese变成DataFrame形式
    # df1.to_excel(resultPath1, sheet_name="gsr频域特征（带行列索引）")