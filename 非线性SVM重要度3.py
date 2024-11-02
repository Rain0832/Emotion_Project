import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
"""
SHAP： 事后解释
甲特征的边际贡献：加入特征甲后，相比未加入时，甲带来的贡献
甲的SHAP值：所有特征组合下，甲的所有边际贡献的平均值
一个样本中各个特征的SHAP值+基线值=模型的最终预测值
"""
# 读取数据（假设数据存储在Excel文件中）
df = pd.read_excel(r"E:\2大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train=X_train[:12]
y_train=y_train[:12]
# 训练SVM模型
svm_model = SVC(C=10,kernel='rbf', gamma=10, decision_function_shape='ovo',probability=True)
svm_model.fit(X_train, y_train)                  # probability=True：启用概率估计，用于后续的SHAP值计算。
# 预测
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 计算SHAP值
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train)  # svm_model.predict_proba：使用模型的概率预测函数。
shap_values = explainer.shap_values(X_test)

# 获取特征重要性
feature_importances = np.abs(shap_values[0]).mean(axis=0)
"""
np.abs(shap_values[0])：计算SHAP值的绝对值，因为SHAP值可以是正的或负的。
.mean(axis=0)：计算每个特征的平均绝对SHAP值，作为特征重要性。
"""
# 打印特征重要性
for i, importance in enumerate(feature_importances):#enumerate(feature_importances) 会生成一个枚举对象，其中每个元素是一个包含两个元素的元组：第一个元素是索引（从0开始），第二个元素是特征重要性值。
    print(f"Feature {i+1}: {importance}")

# 根据特征重要性排序
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
print("排序后：")
# 打印排序后的特征重要性
for i, importance in zip(sorted_indices, sorted_feature_importances):  #
    # zip(sorted_indices, sorted_feature_importances) 会生成一个迭代器，其中每个元素是一个包含两个元素的元组：第一个元素是排序后的特征索引，第二个元素是对应的特征重要性值。
    print(f"Feature {i+1}: {importance}")

# 选择重要性最高的k个特征
k = 10  # 例如选择前10个重要特征
selected_features = sorted_indices[:k]
print("Selected features:", selected_features)
#希望得到SHAP值与图，失败
shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])
"""
100样本
Accuracy: 0.5865921787709497
100%|██████████| 179/179 [13:02<00:00,  4.37s/it]
Feature 1: 0.010007678365628131
Feature 2: 0.010007678365628136
排序后：
Feature 2: 0.010007678365628136
Feature 1: 0.010007678365628131
Selected features: [1 0]

进程已结束，退出代码为 0
"""

"""
Accuracy: 0.5865921787709497
Using 120 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
100%|██████████| 179/179 [15:22<00:00,  5.15s/it]
Feature 1: 0.005784127837119357
Feature 2: 0.0057841278371193764
排序后：
Feature 2: 0.0057841278371193764
Feature 1: 0.005784127837119357
Selected features: [1 0]
"""