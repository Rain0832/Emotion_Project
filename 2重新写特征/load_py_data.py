# import matplotlib.pyplot as plt
# import numpy as np
# data =np.fromfile(r'E:\2大学项目\DEAP\data_preprocessed_python\s09.dat')
# print(data.shape)

import scipy.io
import numpy as np
import pandas as pd
import os

def load_deap_data(data_folder):
    gsr_signal_one = []
    ppg_signal_one = []
    label_final_one = []

    for i in range(1, 33):  # 遍历32个实验者，编号从1到32
        file_name = f's{i:02d}.dat'  # 格式化文件名，例如s01.dat, s02.dat
        file_path = os.path.join(data_folder, file_name)  # 生成文件路径

        if os.path.exists(file_path):  # 如果文件存在
            print(f"Loading data from: {file_path}")
            # 读取二进制 .dat 文件
            data = np.fromfile(file_path, dtype=np.float32)
            # 假设文件包含 40 个视频的数据，每个视频包含 GSR 信号、PPG 信号和标签
            # 每个视频的数据长度为 8064 + 8064 + 4 = 16132 字节
            video_length = 16132
            num_videos = 40
            for video in range(num_videos):
                start_index = video * video_length
                end_index = start_index + video_length
                video_data = data[start_index:end_index]

                gsr_three = video_data[:8064]  # GSR 信号
                gsr_one = gsr_three.reshape(1, -1)  # 调整形状并存储，发现是1x8064
                gsr_signal_one.append(gsr_one)  # 依次在后面添加，最终循环完变成1280x1x8064列表

                ppg_three = video_data[8064:16128]  # PPG 信号
                ppg_one = ppg_three.reshape(1, -1)  # 调整形状并存储，发现是1x8064
                ppg_signal_one.append(ppg_one)  # 依次在后面添加，变成1x（8064x1280）行列表

                label_one = video_data[16128:16132]  # 标签
                label_final_one.append(label_one)

        else:  # 如果文件不存在
            print(f"File not found: {file_path}")

        # 将收集的GSR信号转换为NumPy数组并reshape
    gsr_signal_one = np.array(gsr_signal_one).reshape(-1, 8064)  # 自动计算是几x8064，并变成1280x8064
    label_final_one = np.array(label_final_one)  # 不必.reshape（-1，4），因本来就是二维，最终必为1280x4
    ppg_signal_one = np.array(ppg_signal_one).reshape(-1, 8064)

    return gsr_signal_one, ppg_signal_one, label_final_one  # 返回所有数据和标签

if __name__ == '__main__':
    data_folder = r'E:\2大学项目\DEAP\data_preprocessed_python'
    gsr_signal_one, ppg_signal_one, label_final_one = load_deap_data(data_folder)

    # 打印结果
    print("GSR Signals:", gsr_signal_one.shape)
    print("PPG Signals:", ppg_signal_one.shape)
    print("Labels:", label_final_one.shape)
    #标签
    resultPath1=r"E:\2大学项目\特征提取——数据与程序\py预处理\所有的标签py.xlsx"  #指定excel的路径,文件名：所有的标签
    column_names = ["愉悦度", "唤醒度", "支配度", "喜好"]# 如果 label_final_one 没有列名，创建 DataFrame 时需要指定列名
    df1 = pd.DataFrame(label_final_one,columns=column_names)  #将label_final_one变成DataFrame形式
    df1.to_excel(resultPath1,sheet_name="所有标签",index=False)
      #表格名称：所有标签，无索引
    ##运行程序的时候不要打开其excel文件！！！！
    #GSR
    resultPath2 = r"E:\2大学项目\特征提取——数据与程序\py预处理\所有原GSR数据py（带行列索引）.xlsx"  # 指定excel的路径,
    df2 = pd.DataFrame(gsr_signal_one)  # 将label_final_one变成DataFrame形式
    df2.to_excel(resultPath2, sheet_name="所有GSR数据py（带行列索引）")
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引，“index=False, header=False”应删除
    # #PPG
    resultPath3 = r"E:\2大学项目\特征提取——数据与程序\py预处理\所有原PPG数据py（带行列索引）.xlsx"
    df3 = pd.DataFrame(ppg_signal_one)
    df3.to_excel(resultPath3, sheet_name="所有PPG数据py（带行列索引）")
    # #合并
    resultPath4 = r"E:\2大学项目\特征提取——数据与程序\py预处理\所有原数据总py（带行列索引）.xlsx"
    df4 = pd.concat([df2, df3], ignore_index=True)
    df4.to_excel(resultPath4, sheet_name="所有总数据py（带行列索引）")