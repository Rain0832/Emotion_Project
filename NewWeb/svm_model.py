import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import subprocess

files_to_run = ["a_LoadData.py", "b_preprocess_for_all.py", "c_PPGNormalization.py", "d_feature_extraction.py", "e_combination.py"]

for file in files_to_run:
    subprocess.run(["python", file])

#DEAP的数据
df = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\TrainingData.xlsx",
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
data_new = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\featrue.xlsx",
                     index_col=0, usecols=range(0, 6), nrows=None)
new_data= data_new.iloc[0, :].values
data_std = scaler.transform([new_data]) #标准化
print("-------------------*********---------------------")
# 模型预测
y_pred = model.predict([new_data])
if y_pred==1:
    conclusion = "积极！"
    print("积极!")
else:
    conclusion = "消极！"
    print("消极!")