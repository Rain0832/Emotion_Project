import pyedflib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import fft
import pandas as pd
from scipy.signal import butter, filtfilt
import pywt
"""
针对使用DEAP数据GSR+PPG预处理，保存为2个excel
"""

# file_name = r'E:\2大学项目\final_project_for_memory\所有原始数据\所有原PPG数据.xlsx'
# df = pd.read_excel(file_name, sheet_name=0)
# data = df.to_numpy()

#GSR皮肤电信号预处理
# 读取信号数据
GSR = pd.read_excel(r"E:\2大学项目\final_project_for_memory\0.GSR数据转excel.xlsx",
                     index_col=0, nrows=None)
raw_GSR= GSR.values
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
to_save_GSR = r'E:\2大学项目\final_project_for_memory\1.preprocessed_GSR_final.xlsx'
df_GSR = pd.DataFrame(filtered_GSR)
df_GSR.to_excel(to_save_GSR, index = True)
print("gsr信号数据已保存")


# PPG脉搏波信号预处理
PPG = pd.read_excel(r"E:\2大学项目\final_project_for_memory\0.PPG数据转excel.xlsx",
                     index_col=0, nrows=None)
raw_PPG= PPG.values
fs = 128

# 小波分解
wavelet = 'sym8'  # 使用
level = 8  #

# 进行小波分解
coeffs = pywt.wavedec(raw_PPG, wavelet, level=level)
# coeffs 是一个包含近似系数和细节系数的列表
cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

# 对每一个细节系数应用软阈值
cA_denoised = np.zeros_like(cA8)

# 更新系数
coeffs_denoised = [cA_denoised,cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]

# 使用去噪后的系数重构信号
reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet)
to_save_PPG = r'E:\2大学项目\final_project_for_memory\1.preprocessed_PPG_final.xlsx'
df_PPG = pd.DataFrame(reconstructed_signal)
df_PPG.to_excel(to_save_PPG, index = True)

print("ppg信号数据已保存")
