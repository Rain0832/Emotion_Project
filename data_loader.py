# data_loader.py
import scipy.io
import numpy as np
import pandas as pd
import os

def load_deap_data(data_folder):
    all_data = []  # 该数组存储所有实验者的数据
    all_labels = []  # 该数组存储所有实验者的标签
    gsr_signal_one= []
    ppg_signal_one = []
    label_final_one = []
    for i in range(1, 33):  # 遍历32个实验者，编号从1到32
        file_name = f's{i:02d}.mat'  # 格式化文件名，例如s01.mat, s02.mat
        file_path = os.path.join(data_folder, file_name)  # 生成文件路径
        
        if os.path.exists(file_path):  # 如果文件存在
            print(f"Loading data from: {file_path}")
            mat_data = scipy.io.loadmat(file_path)  # 读取.mat文件
            
            data = mat_data['data']  # 获取数据
            labels = mat_data['labels']  # 获取标签
            
            all_data.append(data)  # 将数据添加到all_data列表
            all_labels.append(labels)   # 将标签添加到all_labels列表
            for video in range(40):
                 gsr_three=data[video,36,:]#三维的，需要变为一维
                 gsr_one = gsr_three.reshape(1, -1)  # 调整形状并存储，发现是1x8064
                 gsr_signal_one.append(gsr_one)#依次在后面添加，最终循环完变成1280x1x8064列表

                 ppg_three = data[video, 38, :]  # 三维的，需要变为一维
                 ppg_one = ppg_three.reshape(1, -1)  # 调整形状并存储，发现是1x8064
                 ppg_signal_one.append(ppg_one)  # 依次在后面添加，变成1x（8064x1280）行列表

                 label_one =labels[video,:]
                 label_final_one.append(label_one)

        else:  # 如果文件不存在
            print(f"File not found: {file_path}")
    # 将收集的GSR信号转换为NumPy数组并reshape
    gsr_signal_one = np.array(gsr_signal_one).reshape(-1, 8064)#自动计算是几x8064，并变成1280x8064
    label_final_one = np.array(label_final_one)#不必.reshape（-1，4），因本来就是二维，最终必为1280x4
    ppg_signal_one = np.array(ppg_signal_one).reshape(-1, 8064)

    all_data = np.array(all_data) # 将all_data列表转换为numpy数组
    all_labels = np.array(all_labels)  # 将all_labels列表转换为numpy数组
    
    return all_data, all_labels, gsr_signal_one,ppg_signal_one,label_final_one  # 返回所有数据和标签

if __name__ == '__main__':
    data_folder = 'data_preprocessed_matlab'  # 加载数据
    all_data, all_labels, gsr_signal_one, ppg_signal_one,label_final_one = load_deap_data(data_folder)
    print(all_data.shape)  # 输出数据的形状
    print(all_labels.shape)  # 输出标签的形状
    print("gsr_signal_one  ",gsr_signal_one.shape)
    print("ppg_signal_one  ",ppg_signal_one.shape)
    print("label_final_one ",label_final_one.shape)
    # #先在cmd中pip install xlsxwriter
    # #标签
    # resultPath1=r"E:\大学项目\特征提取\所有的标签.xlsx"  #指定excel的路径,文件名：所有的标签
    # column_names = ["愉悦度", "唤醒度", "支配度", "喜好"]# 如果 label_final_one 没有列名，创建 DataFrame 时需要指定列名
    # df1 = pd.DataFrame(label_final_one,columns=column_names)  #将label_final_one变成DataFrame形式
    # df1.to_excel(resultPath1,sheet_name="所有标签",index=False)
    #   #表格名称：所有标签，无索引
    #运行程序的时候不要打开其excel文件！！！！
    # #GSR
    # resultPath2 = r"E:\大学项目\所有数据excel\所有原GSR数据（带行列索引）.xlsx"  # 指定excel的路径,
    # df2 = pd.DataFrame(gsr_signal_one)  # 将label_final_one变成DataFrame形式
    # df2.to_excel(resultPath2, sheet_name="所有GSR数据（带行列索引）")
    # ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引，“index=False, header=False”应删除
    # #PPG
    # resultPath3 = r"E:\大学项目\所有数据excel\所有原PPG数据（带行列索引）.xlsx"
    # df3 = pd.DataFrame(ppg_signal_one)
    # df3.to_excel(resultPath3, sheet_name="所有PPG数据（带行列索引）")
    # #合并
    # resultPath4 = r"E:\大学项目\所有数据excel\所有原数据总（带行列索引）.xlsx"
    # df4=pd.concat([df2,df3],ignore_index=True)
    # df4.to_excel(resultPath4, sheet_name="所有总数据（带行列索引）")