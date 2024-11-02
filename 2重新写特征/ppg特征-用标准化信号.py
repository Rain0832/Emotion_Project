import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
N=1280
ppg_all = pd.read_excel(r"E:\2大学项目\2重新写特征\ppg标准化\ppg标准化数据_最小最大值（带行列索引）.xlsx",
                        index_col=0, usecols=range(2300, 3700), skiprows=0, nrows=N)

# ppg = pd.read_excel(r"E:\2大学项目\lmh预处理\finally\ppg_preprocessed_final.xlsx",
#                          index_col=None, usecols=range(0,8064), skiprows=1, nrows=1)
ppg_features = []

# ppg = ppg_all.iloc[876, :]
# ppg = np.array(ppg).flatten()
# # print(ppg)
# plt.plot(ppg)
# plt.show()
# min_distance = 65  # 两个峰最小间距
# min_height = 0.2 # 峰值最小高度
# min_low = 0.85
# peaks, _ = find_peaks(ppg, distance=min_distance, height=min_height)
#
# troughs, _ = find_peaks(-ppg, distance=min_distance, height=-min_low)
# print("peaks",peaks)
# print("troughs",troughs)
# if peaks[0] < troughs[0]:
#     peaks_7 = peaks[1:8]
#     troughs_7 = troughs[0:7]
# else:
#     peaks_7 = peaks[0:7]
#     troughs_7 = troughs[0:7]
# # 从信号中获取峰值的高度
# peak_heights = ppg[peaks_7]
# troughs_height = ppg[troughs_7]
#
# print("peaks_7",peaks_7)
# print("troughs_7",troughs_7)
# plt.plot(ppg)
# plt.show()
# high = peak_heights - troughs_height
# print("high",high)
# aver_high = np.mean(high)
# print("aver_high",aver_high)



for i in range(0, N):
    try:
        ppg = ppg_all.iloc[i, :]
        ppg = np.array(ppg).flatten()
        # print(ppg)
        min_height = 0.2  # 峰值最小高度
        min_low = 0.85
        min_distance = 65
        # plt.plot(ppg)
        # plt.show()
        peaks, _ = find_peaks(ppg, distance=min_distance, height=min_height)
        # print("peaks",peaks)
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
        # print(peaks_7)
        # print("troughs",troughs)
        # print(troughs_7)
        high = peak_heights - troughs_height

        aver_high = np.mean(high)
        ## print(aver_high)

        ppg_features.append(aver_high)
    except Exception as e:
        print(f"在迭代 {"i=",i} 时发生错误: {e}")
    print("i=",i)


print(len(ppg_features))
resultPath3 = r"E:\2大学项目\2重新写特征\PPG1个特征（带行列索引）.xlsx"  # 指定excel的路径,
df3 = pd.DataFrame(ppg_features)  # 将ppg_featurese变成DataFrame形式
df3.to_excel(resultPath3, sheet_name="PPG1个特征（带行列索引）")
