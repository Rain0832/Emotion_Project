import pandas as pd
import numpy as np
"""
特征合并
"""
#合并所有
def combine_feauture():
    gsr_feature = pd.read_excel(r"E:\2大学项目\final_project_for_memory\3.GSR4个特征（带行列索引）.xlsx",
                                index_col=0, usecols=range(0, 5))
    ##usecols=range(0,26)表示读0到25列，index_col=0表示第0列作为行索引,实际有用的1-24列共25个，nrows=None读全部
    ##读数据时，即使打印出索引，真正读时也不会将索引当成真正数据，因此原数据最好保留索引

    ppg_feature = pd.read_excel(r"E:\2大学项目\final_project_for_memory\3.PPG1个特征（带行列索引）.xlsx",
                                index_col=0, usecols=range(0, 2), nrows=None)
    ##先合并gsr+ppg
    gsr_ppg = pd.concat([gsr_feature, ppg_feature], axis=1)
    gsr_ppg.columns = range(1, len(gsr_ppg.columns) + 1)

    #存为excel
    resultPath1 = r"E:\2大学项目\final_project_for_memory\4.原始特征（带行列索引）.xlsx"  # 指定excel的路径,
    df1 = pd.DataFrame(gsr_ppg)
    df1.to_excel(resultPath1, sheet_name="所有原始特征（带行列索引）")
    return 1


if __name__ == '__main__':
    a = 0
    a = combine_feauture()
    if a == 1:
        print("所有原始特征成功合并！！！")
    else:
        print("所有原始特征写入失败")
