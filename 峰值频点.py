from scipy.fftpack import fft, fftfreq  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd
import itertools
#此程序是为了提取正确的峰值频点...
def extract_frequency_domain_features(signal):
    fft_values = abs(fft(signal))  # 取其模值 ，纵坐标 fft_values.shape (8064,)
    #print("fft_values",fft_values)
    half_of_gsr = len(signal) // 2  # =4032 整数除法
    peak_index = np.argmax(fft_values[1:half_of_gsr]) + 1  # 取一半
    freqs = np.fft.fftfreq(len(signal), d=1.0 / 128)  # freqs (8064,) 计算对应横坐标:频率
    #print("freqs",freqs[0:4033])
    half_of_freqs = freqs[0:4032]
    peak_freq = freqs[peak_index]  # 峰值频点

    half_of_fft_values = fft_values[0:4032]
    mean_freq = np.sum(half_of_fft_values*half_of_freqs)/np.sum(half_of_fft_values) # 计算平均频点

    # 计算功率谱密度
    power_spectrum = half_of_fft_values ** 2  # power_spectrum 4032
    # 计算加权频率值的总和和权重的总和
    weighted_sum = np.trapezoid(power_spectrum * half_of_freqs, x=half_of_freqs)
    power_sum = np.trapezoid(power_spectrum, x=half_of_freqs)
    # 计算平均功率谱频率
    avg_power_freq = weighted_sum / power_sum
    return peak_freq, mean_freq, avg_power_freq

if __name__ == '__main__':
    gsr_features = []
    gsr_total = pd.read_excel(r"E:\大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx",
                              index_col=0, usecols=range(0, 8065), nrows=40)
    ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    ##DataFrame形式
    for i in range(0, 40):  # 一行一行提取特征
        gsr = gsr_total.iloc[i, :]  # 第i行所有列
        np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        peak_freq, mean_freq, avg_power_freq = extract_frequency_domain_features(np_gsr)  # 频域特征的第一行:1x3
        #print("peak_freq",i+1,"@@##",peak_freq)
        print("avg_power_freq",i+1,"@@##",mean_freq)

