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
mat_file_name = r'C:\\Users\\23145\\Desktop\\edfreadZip\\s01.mat'
to_save_GSR = r'D:\\Papers\\3. 可穿戴设备项目\\processed_data_1\\s01_GSR.xlsx'
to_save_PPG = r'D:\\Papers\\3. 可穿戴设备项目\\processed_data_2\\s01_PPG.xlsx'
# bdf_file = pyedflib.EdfReader(r'D:\\Papers\\3. 可穿戴设备项目\\data-deap\\s01.bdf')
mat_file = scipy.io.loadmat( mat_file_name )
data = mat_file['data']
# print(data)

#GSR皮肤电信号预处理
# 读取信号数据
raw_GSR = data[ :, 36, : ]
fs = 128

# 设计低通滤波器
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

cutoff_freq = 0.3  # 截止频率 0.3 Hz
filtered_GSR = apply_lowpass_filter(raw_GSR, cutoff_freq, fs)
df_GSR = pd.DataFrame(filtered_GSR)
df_GSR.to_excel(to_save_GSR, index = False)
print("信号数据已保存")


# PPG脉搏波信号预处理
raw_PPG = data[38]
fs = 128

# 小波分解
wavelet = 'db5'  # 使用 Daubechies 5 小波
level = 3  # 三层分解

# 进行小波分解
coeffs = pywt.wavedec(raw_PPG, wavelet, level=level)
# coeffs 是一个包含近似系数和细节系数的列表
cA3, cD3, cD2, cD1 = coeffs

# 基线漂移可能主要集中在 cA3 中，这是最低频的近似系数

# 定义软阈值函数
def soft_thresholding(data, threshold):
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

threshold = np.median(np.abs(cD1)) / 0.6745

# 对每一个细节系数应用软阈值
cA3_denoised = soft_thresholding(cA3, threshold)

# 更新系数
coeffs_denoised = [cA3_denoised, cD3, cD2, cD1]


# 使用去噪后的系数重构信号
reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet)
df_PPG = pd.DataFrame(reconstructed_signal)
df_PPG.to_excel(to_save_PPG, index = False)

print("信号数据已保存")




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






