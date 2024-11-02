import svm-model_new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r"E:\2大学项目\2重新写特征\879个特征样本\879方差标准化.xlsx",
                     index_col=0, usecols=range(0, 6), nrows=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


cat1, cat0 = [], []
for index, single_line in df.iterrows():  # .iterrows() 返回的是一个包含两个元素的元组：索引和行数据。
    feature1 = single_line.iloc[3]
    if single_line['二分类'] == 0:
        cat0.append(feature1)
    else:
        cat1.append(feature1)  # 至此，cat1是标签为1对应的数据

print(len(cat1))
print(cat0)

# plt.scatter(range(524),cat1)
# plt.figure(figsize=(4, 8))
np.random.seed(42)
fruit_weights = [
    cat0,
    cat1,
]
labels = ['cat0', 'cat1' ]
colors = ['peachpuff', 'orange']

fig, ax = plt.subplots()
ax.set_ylabel('values')

bplot = ax.boxplot(fruit_weights,
                   patch_artist=True,  # fill with color
                   tick_labels=labels)  # will be used to label x-ticks

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.title("feature 2")
plt.show()