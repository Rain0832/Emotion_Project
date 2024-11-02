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
file_name = r'D:\\Papers\\3. 可穿戴设备项目\\所有原PPG数据.xlsx'
to_save_GSR = r'D:\\Papers\\3. 可穿戴设备项目\\processed_data_1\\s01_GSR.xlsx'
to_save_PPG = r'D:\\Papers\\3. 可穿戴设备项目\\processed_data_2\\all_PPG.xlsx'
# bdf_file = pyedflib.EdfReader(r'D:\\Papers\\3. 可穿戴设备项目\\data-deap\\s01.bdf')
# mat_file = scipy.io.loadmat( mat_file_name )
# data = mat_file['data']
df = pd.read_excel(file_name, sheet_name=0)
data = df.to_numpy()
# print(data)

# #GSR皮肤电信号预处理
# # 读取信号数据
# raw_GSR = data[ :, 36, : ]
# fs = 128

# # 设计低通滤波器
# def butter_lowpass(cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# def apply_lowpass_filter(data, cutoff_freq, fs, order=5):
#     b, a = butter_lowpass(cutoff_freq, fs, order=order)
#     filtered_data = filtfilt(b, a, data)
#     return filtered_data

# cutoff_freq = 0.3  # 截止频率 0.3 Hz
# filtered_GSR = apply_lowpass_filter(raw_GSR, cutoff_freq, fs)
# df_GSR = pd.DataFrame(filtered_GSR)
# df_GSR.to_excel(to_save_GSR, index = False)
# print("信号数据已保存")


# PPG脉搏波信号预处理
raw_PPG = data[:,:]
print("######",len(raw_PPG[0]))
fs = 128

# 小波分解
wavelet = 'sym8'  # 使用 Daubechies 5 小波
level = 8  # 五层分解

# 进行小波分解
coeffs = pywt.wavedec(raw_PPG, wavelet, level=level)
# coeffs 是一个包含近似系数和细节系数的列表
cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

# 基线漂移可能主要集中在 cA3 中，这是最低频的近似系数

N = 8064
print(N)
# sigma = np.median(np.abs(cD4)) / 0.6745
# threshold = sigma * np.sqrt(2 * np.log(N)) / np.log(6)
# print(threshold)

# # 定义软阈值函数
# def soft_thresholding(data, threshold):
#     return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

# 对每一个细节系数应用软阈值
cA_denoised = np.zeros_like(cA8)
# cA8_denoised = soft_thresholding(cA4, threshold)

# 更新系数
coeffs_denoised = [cA_denoised,cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]

# 使用去噪后的系数重构信号
reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet)
df_PPG = pd.DataFrame(reconstructed_signal)
df_PPG.to_excel(to_save_PPG, index = True)

print("信号数据已保存")

# plt.subplot(4, 2, 1)
# plt.plot(cA3, label='Original cA3', color='b')
# plt.title('Original cA3')
# plt.legend()
# plt.grid(True)

# # 去噪后的 cA3
# plt.subplot(4, 2, 2)
# plt.plot(cA3_denoised, label='Denoised cA3', color='r')
# plt.title('Denoised cA3')
# plt.legend()
# plt.grid(True)

# # 原始 cD3
# plt.subplot(4, 2, 3)
# plt.plot(cD3, label='Original cD3', color='b')
# plt.title('Original cD3')
# plt.legend()
# plt.grid(True)

# # 去噪后的 cD3 (相同，因为未去噪)
# plt.subplot(4, 2, 4)
# plt.plot(cD3, label='Denoised cD3 (Unchanged)', color='g')
# plt.title('Denoised cD3')
# plt.legend()
# plt.grid(True)

# # 原始 cD2
# plt.subplot(4, 2, 5)
# plt.plot(cD2, label='Original cD2', color='b')
# plt.title('Original cD2')
# plt.legend()
# plt.grid(True)

# # 去噪后的 cD2 (相同，因为未去噪)
# plt.subplot(4, 2, 6)
# plt.plot(cD2, label='Denoised cD2 (Unchanged)', color='g')
# plt.title('Denoised cD2')
# plt.legend()
# plt.grid(True)

# # 原始 cD1
# plt.subplot(4, 2, 7)
# plt.plot(cD1, label='Original cD1', color='b')
# plt.title('Original cD1')
# plt.legend()
# plt.grid(True)

# # 去噪后的 cD1 (相同，因为未去噪)
# plt.subplot(4, 2, 8)
# plt.plot(cD1, label='Denoised cD1 (Unchanged)', color='g')
# plt.title('Denoised cD1')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()







# 绘制原始信号和过滤后的信号
# plt.figure(figsize=(12, 6))
 
# 原始信号
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(len(raw_GSR)) / 128, raw_GSR, label='原始信号')
# plt.xlabel('时间 (秒)')
# plt.ylabel('GSR 信号')
# plt.title('原始 GSR 信号')
# plt.legend()

# # 过滤后的信号
# plt.subplot(2, 1, 2)
# plt.plot(t, filtered_GSR, label='过滤后的信号', color='r')
# plt.xlabel('时间 (秒)')
# plt.ylabel('GSR 信号')
# plt.title('过滤后的 GSR 信号')
# plt.legend()

# plt.tight_layout()
# plt.show()