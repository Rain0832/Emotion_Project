# data_loader.py
import scipy.io
import numpy as np
import pandas as pd
import os


def load_deap_data(data_folder):

    gsr_signal_one = []

    for i in range(1, 33):  # 遍历32个实验者，编号从1到32
        file_name = f's{i:02d}.mat'  # 格式化文件名，例如s01.mat, s02.mat
        file_path = os.path.join(data_folder, file_name)  # 生成文件路径

        if os.path.exists(file_path):  # 如果文件存在
            print(f"Loading data from: {file_path}")
            mat_data = scipy.io.loadmat(file_path)  # 读取.mat文件

            data = mat_data['data']  # 获取数据
            labels = mat_data['labels']  # 获取标签
            for video in range(40):
                gsr_three = data[video, 36, :]  # 三维的，需要变为一维
                gsr_one = gsr_three.reshape(-1)  # 调整形状并存储，发现是1x8064
                print(gsr_one)
                gsr_signal_one.append(gsr_one)  # 依次在后面添加，最终循环完变成1280x1x8064列表

        else:  # 如果文件不存在
            print(f"File not found: {file_path}")
    # 将收集的GSR信号转换为NumPy数组并reshape
    gsr_signal_one = np.array(gsr_signal_one).reshape(-1, 8064)  # 自动计算是几x8064，并变成1280x8064

    return gsr_signal_one  # 返回所有数据和标签


if __name__ == '__main__':
    data_folder = 'E:\\2大学项目\DEAP\data_preprocessed_matlab'  # 加载数据
    gsr_signal_one = load_deap_data(data_folder)
    print("gsr_signal_one  ", gsr_signal_one.shape)
    # 运行程序的时候不要打开其excel文件！！！！
    # GSR
    # resultPath2 = r"E:\2大学项目\DEAP\初始可用程序\原始数据对不对\所有原GSR数据检验版（带行列索引）.xlsx"  # 指定excel的路径,
    # df2 = pd.DataFrame(gsr_signal_one)  # 将label_final_one变成DataFrame形式
    # df2.to_excel(resultPath2, sheet_name="sheet1")
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引，“index=False, header=False”应删除
