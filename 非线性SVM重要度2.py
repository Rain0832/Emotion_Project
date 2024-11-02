import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
"""
置换重要性:（Permutation Importance）：通过随机打乱某个特征的值，观察模型性能的变化来评估特征重要性。
"""
# X.shape = (892, 50)
df = pd.read_excel(r"E:\2大学项目\二分类特征+标签\筛选后特征+标签2（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 训练SVM模型
svm_model = SVC(C=10,kernel='rbf', gamma=10, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 计算置换重要性
result = permutation_importance(svm_model, X_test, y_test, n_repeats=30, random_state=42)

# 获取特征重要性
feature_importances = result.importances_mean

# 打印特征重要性
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance}")

# 根据特征重要性排序
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]

# 打印排序后的特征重要性
for i, importance in zip(sorted_indices, sorted_feature_importances):
    print(f"Feature {i+1}: {importance}")

# 选择重要性最高的k个特征
k = 10  # 例如选择前10个重要特征
selected_features = sorted_indices[:k]
print("Selected features:", selected_features)