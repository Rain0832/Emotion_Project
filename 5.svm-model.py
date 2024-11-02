import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import subprocess

files_to_run = ["0读数据txt.py", "1.preprocess_for_all.py", "2.ppg归一化.py", "3.feature_extraction.py", "4.combination.py"]

for file in files_to_run:
    subprocess.run(["python", file])


#DEAP的数据
df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)
# 将数据分为特征和目标变量
X = df.iloc[:, :-1].values  # 特征 ()
y = df.iloc[:, -1].values  # 目标
# 特征标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 构建并训练SVM模型
model = SVC(C=9.9, kernel='rbf', gamma=10.5, decision_function_shape='ovo')
model.fit(X_std, y)

# 加载新采样的样本
data_new = pd.read_excel(r"E:\2大学项目\final_project_for_memory\4.原始特征（带行列索引）.xlsx",
                     index_col=0, usecols=range(0, 6), nrows=None)
new_data= data_new.iloc[0, :].values
data_std = scaler.transform([new_data]) #标准化
print("-------------------*********---------------------")
# 模型预测
y_pred = model.predict([new_data])
if y_pred==1:
    print("积极!")
else:
    print("消极!")



