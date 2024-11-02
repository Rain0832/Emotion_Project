import pandas as pd
import numpy as np

#合并所有
def combine_feauture_and_labels():
    gsr_feature = pd.read_excel(r"E:\大学项目\特征提取\GSR所有特征2（带行列索引）.xlsx",
                                index_col=0, usecols=range(0, 26), nrows=None)
    ##usecols=range(0,26)表示读0到25列，index_col=0表示第0列作为行索引,实际有用的1-24列共25个，nrows=None读全部
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引
    ##DataFrame形式 [1280 rows x 25 columns]
    #print(gsr_feature)
    ppg_feature = pd.read_excel(r"E:\大学项目\特征提取\PPG所有特征2（带行列索引）.xlsx",
                                index_col=0, usecols=range(0, 26), nrows=None)
    labels = pd.read_excel(r"E:\大学项目\特征提取\二分类原始标签（带索引）.xlsx", index_col=0)
    #先合并gsr+ppg
    gsr_ppg = pd.concat([gsr_feature, ppg_feature], axis=1)
    gsr_ppg.columns = range(1, len(gsr_ppg.columns) + 1)
    #再合并label
    result = pd.concat([gsr_ppg, labels], axis=1)
    #存为excel
    resultPath1 = r"E:\大学项目\二分类特征+标签\所有原始特征+标签2（带行列索引）.xlsx"  # 指定excel的路径,
    df1 = pd.DataFrame(result)
    df1.to_excel(resultPath1, sheet_name="所有原始特征+标签2（带行列索引）")
    return 1

#筛选所有
def select_feauture_and_labels():
    previous_data = pd.read_excel(r"E:\大学项目\二分类特征+标签\所有原始特征+标签2（带行列索引）.xlsx",
                                  index_col=0, nrows=None)
    selected_data = []
    abandoned = 0  # 舍弃的样本数
    for index, single_line in previous_data.iterrows():  # 将愉悦度分为两类：6-9为正性（1），1-4为负性（0）
        df_label = single_line["二分类"]
        if df_label > 6:
            pleasure = 1
        elif df_label < 4:
            pleasure = 0
        else:
            pleasure = -100
            abandoned += 1
            continue
        single_line["二分类"] = pleasure
        if pleasure != -100:  # 若pleasure==100，则single_line虽被改值为100，但不会被写入
            selected_data.append(single_line)

    print("应该舍弃", abandoned, "个样本")
    resultPath2 = r"E:\大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx"  # 指定excel的路径,
    df2 = pd.DataFrame(selected_data)
    df2.reset_index(drop=True, inplace=True)  # 重置行索引，丢弃旧的索引
    df2.to_excel(resultPath2, sheet_name="筛选后特征+标签2（带行列索引）")
    return 1


if __name__ == '__main__':
    a, b = 0, 0
    a = combine_feauture_and_labels()
    b = select_feauture_and_labels()
    if a == 1:
        print("所有原始特征+标签成功写入！！！")
    else:
        print("所有原始特征+标签写入失败")
    if b == 1:
        print("筛选后特征+标签成功写入！！！")
    else:
        print("筛选后特征+标签写入失败")
