import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
from flask import Flask, render_template
from io import BytesIO
import base64
from svm_model import conclusion

# 设置使用非交互式后端
matplotlib.use('Agg')
app = Flask(__name__)

# 加载数据
df = pd.read_excel(r"Data.xlsx",index_col=0, usecols=range(0, 52), nrows=None)

columns_to_drop = ['二分类']
X = df.iloc[:, :1]  # 选择第一列作为特征
y = df[columns_to_drop].values.ravel()  # 标签

# 划分数据
X_train, y_train = X[:600], y[:600]
X_test, y_test = X[600:], y[600:]

# 创建SVM模型
svm_clf = Pipeline([
    ('first', StandardScaler()),  # 数据标准化
    ('second', SVC(kernel='rbf', C=1, random_state=64, gamma=0.1))  # SVM模型
])
# 训练模型
svm_clf.fit(X_train, y_train)

# 交叉验证
y_pred = cross_val_predict(svm_clf, X_train, y_train)  # 预测的y值
result = confusion_matrix(y_train, y_pred)  # 计算混淆矩阵

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/consult')
def consult():
    return render_template('consult.html')

@app.route('/test')
def test():
    return render_template('test.html',conclusion=conclusion)

@app.route('/project')
def show_confusion_matrix():  # 我将函数名改成了show_confusion_matrix
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

    # 保存图像到内存
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # 关闭图像
    img.seek(0)

    # 将图像数据转换为base64编码
    image_data = base64.b64encode(img.getvalue()).decode('utf-8')

    # 将图像传递给模板
    return render_template('project.html', image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True)
