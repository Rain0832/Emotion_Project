import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
from flask import Flask, render_template
from io import BytesIO
import base64
from svm_model import conclusion # 导入并运行svm文件程序
# from svm-finally_newdata import conclusion
from flask_socketio import SocketIO, emit
import threading
import time


# 先 pip install flask flask-socketio
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 设置使用非交互式后端
matplotlib.use('Agg')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

ppg_line = []
def generate_random_numbers():
    PPG= pd.read_csv(r"D:\桌面\Gitfile\NewWeb\Data\TSensorData.txt", sep=',', header=0, usecols=[2])
    length = len(PPG)
    i = 1
    while i:
        if i >= length:
            break
        else:
            num=PPG.iloc[i,0]
            print("num", num)
            ppg_line.append(num)
            if len(ppg_line) > 15:  # 限制列表大小，只保留最新的10个数据点
                ppg_line.pop(0)
            socketio.emit('new random number', {'number': num})
            time.sleep(0.5)
        i = i + 1

# 加载数据
df = pd.read_excel(r"D:\桌面\Gitfile\NewWeb\Data\TrainingData.xlsx",
                     index_col=0, usecols=range(0, 7), nrows=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建并训练SVM模型
model = SVC(C=9.9, kernel='rbf', gamma=10.5, decision_function_shape='ovo')
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)
result= confusion_matrix(y_test, y_pred)

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
    # 创建雷达图

    return render_template('test.html', conclusion=conclusion)


@socketio.on('connect')
def on_connect():
    threading.Thread(target=generate_random_numbers).start()


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


@app.route('/sign')
def sign():
    return render_template('sign.html')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
