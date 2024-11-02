import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# 加载数据
# All_Data.xlsx：892*51
# 有效样本量892例（除去中性效价）；
# 每例样本有效特征50个（皮肤电和脉搏波各25个），每例样本分类标签1个（0为负性，1为正性）
data = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\4特征+哈工大\4特征+哈工大剔除\4特征+特征9\4特征+特征9.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)

# 数据预处理
# 将数据分为特征和目标变量
X = data.iloc[:, :-1].values  # 特征 (892, 50)
y = data.iloc[:, -1].values  # 目标 (892,1)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# 892 * 0.3 = 267.6 ≈ 268；所以测试集样本为268例


# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建并训练SVM模型
model = SVC(C=9.9, kernel='rbf', gamma=10.5, decision_function_shape='ovo')
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)


# 评估模型
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
result= confusion_matrix(y_test, y_pred)
# #
# y_pred = model.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)
# recall = recall_score(y_train, y_pred)
# f1 = f1_score(y_train, y_pred)
# report = classification_report(y_train, y_pred)
# result= confusion_matrix(y_train, y_pred)

print(f'Accuracy: {accuracy:.4f}', f'recall: {recall:.4f}',f'f1: {f1:.4f}')
print("confusion metrix:\n",result)

print('Classification Report:')
print(report)

def show_confusion_matrix(result):
    plt.figure(figsize=(8, 6))
    sns.heatmap(result,
                annot=True,  # 在热图上显示值
                linewidths=.7,
                fmt='d',  # 显示整数格式
                cmap='YlOrRd',  # 从黄色到红色的渐变
                xticklabels=['Predicted: 0', 'Predicted: 1'],
                yticklabels=['Actual: 0', 'Actual: 1'],
                linecolor='green')  # 设置边框颜色为绿色

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


show_confusion_matrix(result)
