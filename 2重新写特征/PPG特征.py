import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ppg_all = pd.read_excel(r"E:\2大学项目\lmh预处理\finally\preprocessed_PPG_final.xlsx",
                        index_col=0, usecols=range(2300, 3700), skiprows=0, nrows=41)
# ppg = pd.read_excel(r"E:\2大学项目\lmh预处理\finally\ppg_preprocessed_final.xlsx",
#                          index_col=None, usecols=range(0,8064), skiprows=1, nrows=1)
ppg_features = []
ppg = ppg_all.iloc[39, :]
ppg = np.array(ppg).flatten()
# print(ppg)
min_distance = 100  # 两个峰最小间距
min_height = 1000  # 峰值最小高度
peaks, _ = find_peaks(ppg, distance=min_distance, height=min_height)

troughs, _ = find_peaks(-ppg, distance=100, height=-min_height)
if peaks[0] < troughs[0]:
    peaks_7 = peaks[1:8]
    troughs_7 = troughs[0:7]
else:
    peaks_7 = peaks[0:7]
    troughs_7 = troughs[0:7]


# 从信号中获取峰值的高度
peak_heights = ppg[peaks_7]
troughs_height = ppg[troughs_7]
print("peaks",peaks)
print("peaks_7",peaks_7)
print("troughs",troughs)
print("troughs_7",troughs_7)

plt.plot(ppg)
plt.show()
high = peak_heights - troughs_height
print("high",high)
aver_high = np.mean(high)
print("aver_high",aver_high)




    # print(i+1)



for i in range(0, 1280):
    try:
        ppg = ppg_all.iloc[i, :]
        ppg = np.array(ppg).flatten()
        # print(ppg)
        min_distance = 100  # 两个峰最小间距
        min_height = 2000  # 峰值最小高度
        peaks, _ = find_peaks(ppg, distance=min_distance, height=min_height)
        # print("peaks",peaks)
        troughs, _ = find_peaks(-ppg, distance=100, height=-min_height)
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
        # print(aver_high)
        # plt.plot(ppg)
        # plt.show()

        ppg_features.append(aver_high)
    except Exception as e:
        print(f"在迭代 {i} 时发生错误: {e}")
    # print(i+1)


print(len(ppg_features))
resultPath3 = r"E:\2大学项目\2重新写特征\PPG1个特征（带行列索引）.xlsx"  # 指定excel的路径,
df3 = pd.DataFrame(ppg_features)  # 将ppg_featurese变成DataFrame形式
df3.to_excel(resultPath3, sheet_name="PPG1个特征（带行列索引）")
