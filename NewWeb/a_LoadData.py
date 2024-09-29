import pandas as pd

# 读取txt文件
file_path = r'D:\桌面\Gitfile\NewWeb\Data\TSensorData.txt'  # txt文件路径
data = pd.read_csv(file_path)

# 提取需要的列
extracted_data1 = data[['GSR']]
extracted_data2 = data[['PPG']]

# 将数据写入Excel
output_file1 = r'D:\桌面\Gitfile\NewWeb\Data\rawGSR.xlsx'
output_file2 = r'D:\桌面\Gitfile\NewWeb\Data\rawPPG.xlsx'

extracted_data1.to_excel(output_file1, index=True)
extracted_data2.to_excel(output_file2, index=True)

print(f"数据已成功写入 {output_file1} {output_file2}")