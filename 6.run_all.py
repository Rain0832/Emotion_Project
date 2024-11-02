import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
files_to_run = ["0读数据txt.py", "1.preprocess_for_all.py", "2.ppg归一化.py", "3.feature_extraction.py", "4.combination.py", "5.svm-model.py"]

for file in files_to_run:
    subprocess.run(["python", file])

# 读取PPG数据
ppg_wave = pd.read_excel(r"E:\2大学项目\final_project_for_memory\1.preprocessed_PPG_final.xlsx",
                         index_col=0, usecols=range(0, 300), header=0)
ppg_wave = ppg_wave.values.flatten()

# 读取GSR数据
gsr_wave = pd.read_excel(r"E:\2大学项目\final_project_for_memory\1.preprocessed_GSR_final.xlsx",
                         index_col=0, usecols=range(0, 500), header=0)
gsr_wave = gsr_wave.values.flatten()

# 创建一个图形窗口和两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(gsr_wave)
ax1.set_title("GSR waveform")

ax2.plot(ppg_wave)
ax2.set_title("PPG waveform")

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()



#分别依次展示波形
# ppg_wave = pd.read_excel(r"E:\2大学项目\final_project_for_memory\1.preprocessed_PPG_final.xlsx",
#                    index_col=0,usecols=range(0,300), header=0)
# ppg_wave =ppg_wave.values.flatten()
# plt.plot(ppg_wave)
# plt.title("PPG waveform")
# plt.show()
#
# gsr_wave = pd.read_excel(r"E:\2大学项目\final_project_for_memory\1.preprocessed_GSR_final.xlsx",
#                    index_col=0,usecols=range(0,300), header=0)
# gsr_wave =gsr_wave.values.flatten()
# plt.plot(gsr_wave)
# plt.title("GSR waveform")
# plt.show()