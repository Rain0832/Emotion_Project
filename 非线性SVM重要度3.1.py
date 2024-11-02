import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
# 读取数据
df = pd.read_excel(r"E:\大学项目\预处理后数据的标准化\标准化数据_方差均值（带行列索引）.xlsx",
                   index_col=0, usecols=range(0, 52), nrows=None)
X = df.iloc[:, :-45].values
y = df.iloc[:, -1].values
new_df = df.iloc[:, :-45]
# print(X.shape) (892, 6)
# print(df.head())
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 限制训练集大小（可选）
X_train = X_train[:30]
y_train = y_train[:30]

# 训练SVM模型
svm_model = SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovo', probability=True)
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 计算SHAP值
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)  #892x0.2=178.4-》179
# print("shap value:",shap_values)
# 提取每个类别的 SHAP 值
shap_values_class_0 = shap_values[:, :, 0]  # 提取每个类别的SHAP值。shap_values 是一个三维数组，其中第三维对应类别。
print("shap_values_class_0", len(shap_values_class_0))
shap_values_class_1 = shap_values[:, :, 1]
print("shap_values_class_1", len(shap_values_class_1))
# 计算每个类别的特征贡献度
importance_class_0 = np.abs(shap_values_class_0).mean(axis=0)
importance_class_1 = np.abs(shap_values_class_1).mean(axis=0)
importance_df = pd.DataFrame({
    '类别0': importance_class_0,
    '类别1': importance_class_1,
}, index=new_df.columns)
# 根据Type和Type_encoded对照表修改列名
type_mapping = {
    0: '类型A',
    1: '类型B',
}
importance_df.columns = [type_mapping[int(col.split('类别')[1])] for col in importance_df.columns]
print(importance_df)
# # 设置支持中文字符的字体
# font_path = 'C:/Windows/Fonts/simsun.ttc'  # 例如，使用SimSun字体
# prop = fm.FontProperties(fname=font_path)
#
# # 检查数据分布
# for feature in new_df.columns:
#     sns.histplot(df[feature][y == 0], color='blue', label='类别0', alpha=0.5)
#     sns.histplot(df[feature][y == 1], color='red', label='类别1', alpha=0.5)
#     plt.title(feature, fontproperties=prop)
#     plt.legend()
#     plt.show()
#
# # 可视化SHAP值
# shap.summary_plot(shap_values, X_test, feature_names=new_df.columns)

# 可视化类别0的特征重要性
shap.summary_plot(shap_values[:,:,0], X_test, feature_names=new_df.columns)
# # 可视化类别1的特征重要性
# shap.summary_plot(shap_values[:,:,1], X_test, feature_names=new_df.columns)
# 可视化类别0和类别1的SHAP值的条形图
shap.bar_plot(shap_values, X_test)

