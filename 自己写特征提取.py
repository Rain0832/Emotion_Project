from scipy.fftpack import fft  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd

gsr = pd.read_excel(r"E:\大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx",
                    index_col=0, usecols=range(0, 8065), nrows=1)
##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
##DataFrame形式
np_gsr = np.array(gsr).reshape(-1) # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组

fft_values = abs(fft(np_gsr))  #取其模值 ，纵坐标 fft_values.shape (8064,)
print("fft_values",fft_values)

half_of_gsr = len(np_gsr)//2 # =4032 整数除法
peak_index = np.argmax(fft_values[1:half_of_gsr])+1  #取一半
print("peak_index", peak_index)
freqs = np.fft.fftfreq(len(np_gsr), d=1.0/128)# freqs (8064,) 计算对应横坐标:频率
print("freqs", freqs)
peak_freq = freqs[peak_index]# 峰值频点
print("peak_freq", peak_freq)










# a=np.array([[[[[1,2,3,5,67,7]]]]]).reshape(-1)
# print("a",a)
# print("a.shape",a.shape)
