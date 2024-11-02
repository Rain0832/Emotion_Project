from scipy.fftpack import fft  # 导入用于快速傅里叶变换的库
import numpy as np  # 导入NumPy库
import pandas as pd
# 提取时域特征：22个
def extract_time_domain_features(signal):

    peak_to_peak = np.ptp(signal)  # 计算峰峰值
    std_dev = np.std(signal)  # 计算标准差
    mean = np.mean(signal)  # 计算均值
    energy = np.sum(signal**2)  # 计算能量
    quartile_range = np.percentile(signal, 75) - np.percentile(signal, 25)  # 计算四分位距
    zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()  # 计算过零点数量
    diff_signal = np.diff(signal)  # 计算一阶差分信号
    abs_diff_mean = np.mean(np.abs(diff_signal))  # 计算一阶差分绝对值均值
    diff_quartile_range = np.percentile(diff_signal, 75) - np.percentile(diff_signal, 25)  # 计算一阶差分四分位距
    diff_std_dev = np.std(diff_signal)  # 计算一阶差分标准差
    diff_peak_to_peak = np.ptp(diff_signal)  # 计算一阶差分峰峰值
    diff2_signal = np.diff(diff_signal)  # 计算二阶差分信号
    abs_diff2_mean = np.mean(np.abs(diff2_signal))  # 计算二阶差分绝对值均值
    diff2_quartile_range = np.percentile(diff2_signal, 75) - np.percentile(diff2_signal, 25)  # 计算二阶差分四分位距
    diff2_std_dev = np.std(diff2_signal)  # 计算二阶差分标准差
    diff2_peak_to_peak = np.ptp(diff2_signal)  # 计算二阶差分峰峰值
    sq_signal = signal ** 2  # 计算平方信号
    sq_mean = np.mean(sq_signal)  # 计算平方均值
    sq_quartile_range = np.percentile(sq_signal, 75) - np.percentile(sq_signal, 25)  # 计算平方四分位距
    sq_std_dev = np.std(sq_signal)  # 计算平方标准差
    sq_peak_to_peak = np.ptp(sq_signal)  # 计算平方峰峰值
    cu_signal = signal ** 3  # 计算立方信号
    cu_mean = np.mean(cu_signal)  # 计算立方均值
    cu_quartile_range = np.percentile(cu_signal, 75) - np.percentile(cu_signal, 25)  # 计算立方四分位距
    cu_std_dev = np.std(cu_signal)  # 计算立方标准差
    cu_peak_to_peak = np.ptp(cu_signal)  # 计算立方峰峰值
    return (peak_to_peak, std_dev, mean, energy, quartile_range, zero_crossings,
            abs_diff_mean, diff_quartile_range, diff_std_dev, diff_peak_to_peak,
            abs_diff2_mean, diff2_quartile_range, diff2_std_dev, diff2_peak_to_peak,
            sq_mean, sq_quartile_range, sq_std_dev, sq_peak_to_peak,cu_mean,
            cu_quartile_range, cu_std_dev, cu_peak_to_peak)

# 提取频域特征：3个
def extract_frequency_domain_features(signal):
    fft_values = abs(fft(signal))  # 取其模值 ，纵坐标 fft_values.shape (8064,)
    #print("fft_values",fft_values)
    half_of_gsr = len(signal) // 2  # =4032 整数除法
    peak_index = np.argmax(fft_values[1:half_of_gsr]) + 1  # 取一半
    freqs = np.fft.fftfreq(len(signal), d=1.0 / 128)  # freqs (8064,) 计算对应横坐标:频率
    half_of_freqs = freqs[0:4032]
    peak_freq = freqs[peak_index]  # 峰值频点

    half_of_fft_values = fft_values[0:4032]
    mean_freq = np.sum(half_of_fft_values*half_of_freqs)/np.sum(half_of_fft_values) # 计算平均频点

    # 计算功率谱密度
    power_spectrum = half_of_fft_values ** 2  # power_spectrum 4032
    # 计算加权频率值的总和和权重的总和
    weighted_sum = np.trapezoid(power_spectrum * half_of_freqs, x=half_of_freqs)
    power_sum = np.trapezoid(power_spectrum, x=half_of_freqs)
    avg_power_freq = weighted_sum / power_sum  # 计算平均功率谱频率
    return peak_freq, mean_freq, avg_power_freq

## 二分类标签
def pleasure_label_to_excel():
    pleasure_labels = []
    df_labels = pd.read_excel(r"E:\大学项目\特征提取\所有的标签.xlsx", usecols=["唤醒度"])
    # index_col = 0不要行索引,只读唤醒度列
    # 最好保留标签
    df_labels = df_labels.reset_index(drop=True)  # 不要原索引，并重置索引，1280x1
    print(df_labels)
    # 二分类标签存excel文件
    df_labels["唤醒度"] = pd.to_numeric(df_labels["唤醒度"], errors='coerce')
    # 转成数字形式，errors='coerce': 当转换过程中遇到无法转换为数值的数据时,Pandas将这些数据转换为 NaN（不是数字）
    abandoned = 0  # 舍弃的样本数
    for df_label in df_labels["唤醒度"]:  # 将愉悦度分为两类：6-9为正性（1），1-4为负性（0）
        if df_label > 6:
            pleasure = 1
        elif df_label < 4:
            pleasure = 0
        else:
            pleasure = -100
            abandoned = abandoned + 1

        pleasure_labels.append(df_label)
    print("应该舍弃", abandoned, "个样本")
    resultPath1 = r"E:\大学项目\特征提取\二分类原始标签（带索引）.xlsx"  # 将二分类标签导出excel
    column_names = ["二分类"]  # 列名称为二分类
    df1 = pd.DataFrame(pleasure_labels, columns=column_names)
    df1.to_excel(resultPath1, sheet_name="二分类原始标签（带索引）")
    return 1
# GSR特征写入excel
def gsr_features_to_excel():
    ##gsr
    gsr_features = []
    gsr_total = pd.read_excel(r"E:\大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx",
                              index_col=0, usecols=range(0, 8065), nrows=None)
    ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    ##DataFrame形式
    for i in range(0, 1280):  # 一行一行提取特征
        gsr = gsr_total.iloc[i, :]  # 第i行所有列
        np_gsr = np.array(gsr).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        peak_freq, mean_freq, avg_power_freq = extract_frequency_domain_features(np_gsr)  # 频域特征的第一行:1x3
        (peak_to_peak, std_dev, mean, energy, quartile_range, zero_crossings,
         abs_diff_mean, diff_quartile_range, diff_std_dev, diff_peak_to_peak,
         abs_diff2_mean, diff2_quartile_range, diff2_std_dev, diff2_peak_to_peak,
         sq_mean, sq_quartile_range, sq_std_dev, sq_peak_to_peak, cu_mean,
         cu_quartile_range, cu_std_dev, cu_peak_to_peak) = extract_time_domain_features(np_gsr)

        single_line = (peak_to_peak, std_dev, mean, energy, quartile_range, zero_crossings,
                       abs_diff_mean, diff_quartile_range, diff_std_dev, diff_peak_to_peak,
                       abs_diff2_mean, diff2_quartile_range, diff2_std_dev, diff2_peak_to_peak,
                       sq_mean, sq_quartile_range, sq_std_dev, sq_peak_to_peak, cu_mean,
                       cu_quartile_range, cu_std_dev, cu_peak_to_peak, peak_freq, mean_freq, avg_power_freq)
        gsr_features.append(single_line)
    resultPath2 = r"E:\大学项目\特征提取\GSR所有特征2（带行列索引）.xlsx"  # 指定excel的路径,
    df2 = pd.DataFrame(gsr_features)  # 将gsr_featurese变成DataFrame形式
    df2.to_excel(resultPath2, sheet_name="GSR所有特征2（带行列索引）")
    # 返回所有特征的列表
    return 1
#ppg特征写入excel
def ppg_features_to_excel():
    ##ppg
    ppg_features = []
    ppg_total = pd.read_excel(r"E:\大学项目\所有数据excel\所有原PPG数据（带行列索引）.xlsx",
                              index_col=0, usecols=range(0, 8065), nrows=None)
    ##usecols=range(0,8065)表示读0到8064列，index_col=0表示第0列作为行索引，nrows=4读前4行
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    ##DataFrame形式
    for i in range(0, 1280):  # 一行一行提取特征
        ppg = ppg_total.iloc[i, :]  # 第i行所有列
        np_ppg = np.array(ppg).reshape(-1)  # print出来是np_gsr.shape (8064,)表示一维数组，(1, 8064)是二维数组
        # 将DataFrame形式转换成numpy数组形式
        peak_freq, mean_freq, avg_power_freq = extract_frequency_domain_features(np_ppg)  # 频域特征的第一行:1x3
        (peak_to_peak, std_dev, mean, energy, quartile_range, zero_crossings,
         abs_diff_mean, diff_quartile_range, diff_std_dev, diff_peak_to_peak,
         abs_diff2_mean, diff2_quartile_range, diff2_std_dev, diff2_peak_to_peak,
         sq_mean, sq_quartile_range, sq_std_dev, sq_peak_to_peak, cu_mean,
         cu_quartile_range, cu_std_dev, cu_peak_to_peak) = extract_time_domain_features(np_ppg)

        single_line = (peak_to_peak, std_dev, mean, energy, quartile_range, zero_crossings,
                       abs_diff_mean, diff_quartile_range, diff_std_dev, diff_peak_to_peak,
                       abs_diff2_mean, diff2_quartile_range, diff2_std_dev, diff2_peak_to_peak,
                       sq_mean, sq_quartile_range, sq_std_dev, sq_peak_to_peak, cu_mean,
                       cu_quartile_range, cu_std_dev, cu_peak_to_peak, peak_freq, mean_freq, avg_power_freq)
        ppg_features.append(single_line)
    resultPath3 = r"E:\大学项目\特征提取\PPG所有特征2（带行列索引）.xlsx"  # 指定excel的路径,
    df3 = pd.DataFrame(ppg_features)  # 将ppg_featurese变成DataFrame形式
    df3.to_excel(resultPath3, sheet_name="PPG所有特征2（带行列索引）")
    return 1

if __name__ == '__main__':
    #先pip install xlrd,pip install openpyxl
    a,b,label = 0,0,0
    # label = pleasure_label_to_excel()
    a = gsr_features_to_excel()
    b = ppg_features_to_excel()
    if label == 1:
        print("二分类标签成功写入excel！")
    else:
        print("二分类标签写入excel失败")
    if a == 1:
        print("gsr特征成功写入excel！")
    else:
        print("gsr特征写入excel失败")
    if b == 1:
        print("ppg特征成功写入excel！")
    else:
        print("ppg特征写入excel失败")




