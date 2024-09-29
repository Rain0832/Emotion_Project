import pandas as pd
import numpy as np


#合并所有
def combine_feauture():
    gsr_feature = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\b_GSRFeature.xlsx",
                                index_col=0, usecols=range(0, 5))

    ppg_feature = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\b_PPGFeature.xlsx",
                                index_col=0, usecols=range(0, 2), nrows=None)
    
    # 合并gsr+ppg
    gsr_ppg = pd.concat([gsr_feature, ppg_feature], axis=1)
    gsr_ppg.columns = range(1, len(gsr_ppg.columns) + 1)

    #存为excel
    resultPath1 = r"D:\桌面\Gitfile\NewWeb\Data\Features.xlsx"  # 指定excel的路径,
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
