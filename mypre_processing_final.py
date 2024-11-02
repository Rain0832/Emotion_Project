import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import fft
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy.io
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib
import pywt

# 全局设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 打开BDF文件
file_name = r'E:\2大学项目\所有数据excel\所有原PPG数据（带行列索引）.xlsx'

df = pd.read_excel(file_name,index_col=0,header=0)
# print(df)
data = df.to_numpy()
# print(data)

# PPG脉搏波信号预处理
raw_PPG = data[:,:]
# print("######",len(raw_PPG[0]))
fs = 128

# 小波分解
wavelet = 'sym8'  # 使用 Daubechies 5 小波
level = 8  # 五层分解
new_signal = []
for i in range(1280):

    # 进行小波分解
    coeffs = pywt.wavedec(raw_PPG[i], wavelet, level=level)
    # coeffs 是一个包含近似系数和细节系数的列表
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # print(coeffs,len(coeffs))
    # 基线漂移可能主要集中在 cA3 中，这是最低频的近似系数

    N = 8064
    # print(N)

    # 对每一个细节系数应用软阈值
    cA_denoised = np.zeros_like(cA8)
    # cA8_denoised = soft_thresholding(cA4, threshold)

    # 更新系数
    coeffs_denoised = [cA_denoised,cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
    reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet)
    new_signal.append(reconstructed_signal)
to_save_PPG = r'E:\2大学项目\my预处理\\preprocessed_PPG_final.xlsx'
print(new_signal)
# 使用去噪后的系数重构信号
df_PPG = pd.DataFrame(new_signal)
print(df_PPG)
df_PPG.to_excel(to_save_PPG, index = True)
print("信号数据已保存")
