# model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data_loader import load_deap_data
from feature_extractionfinally import prepare_feature_dataset

# 加载DEAP数据集：调用自定义函数
data_folder = 'data_preprocessed_matlab'
all_data, all_labels = load_deap_data(data_folder)

# 准备特征数据集：调用自定义函数
features, labels = prepare_feature_dataset(all_data, all_labels)

# 检查标签的分布：得知训练集/测试集中的正负比例（正常应该在1：1左右）
unique, counts = np.unique(labels, return_counts=True)
print("Labels distribution:", dict(zip(unique, counts)))

# 标签二值化（已经在 prepare_feature_dataset 中处理）

# 分割数据集（20%作为测试集）
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

# 检查训练集中标签的分布
unique_train, counts_train = np.unique(y_train, return_counts=True)
print("Training labels distribution:", dict(zip(unique_train, counts_train)))

# 标准化特征（处理量纲不同带来的影响）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 处理数据不平衡（如果标签中大部分是正或者大部分是负）（手动调整训练集）
if len(counts_train) < 2:
    raise ValueError("Training set contains only one class. Adjust your data preprocessing steps.")

count_class_0, count_class_1 = counts_train
df_train = pd.DataFrame(X_train)
df_train['label'] = y_train
df_class_0 = df_train[df_train['label'] == 0]
df_class_1 = df_train[df_train['label'] == 1]

# 平衡类别数量
df_class_0_over = df_class_0.sample(count_class_1, replace=True)
df_train_balanced = pd.concat([df_class_0_over, df_class_1], axis=0)

X_train_balanced = df_train_balanced.drop('label', axis=1).values
y_train_balanced = df_train_balanced['label'].values

# 训练SVM模型（调用SVC函数）
model = SVC(kernel='rbf', random_state=42,gamma=0.1)
model.fit(X_train_balanced, y_train_balanced)

# 预测
#y_pred = model.predict(X_test)
y_pred = model.predict(X_train)

# 评估模型
print(confusion_matrix(y_test, y_pred)) # 混淆矩阵
print(classification_report(y_test, y_pred)) # 分类报告

# 画出真实值和预测值的分类结果图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.6, label='True Labels')
plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.6, label='Predicted Labels')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('True vs Predicted Labels')
plt.show()
