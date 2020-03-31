from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
import time
from sklearn.model_selection import train_test_split
import matplotlib
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#  将数据归一化到0~1之间，80%作为训练集，其余作为测试集
#active-->label
data_label_csv = pd.read_csv('milan-data-active-sensor-train.csv',usecols=[4])
#sensor-->input
data_input_csv = pd.read_csv('milan-data-active-sensor-train.csv',usecols=[5])
# print(data_csv)
# data_csv = data_csv.drop_duplicates()
# print(data_csv)
# print(data_csv.size)

#数据预处理
data_label_csv = data_label_csv.dropna() #去掉na数据
datasetY = data_label_csv.values
datasetY = datasetY.astype('int32') #整数标签
print(type(datasetY))
print(datasetY.shape)

data_input_csv = data_input_csv.dropna() #去掉na数据
datasetX = data_input_csv.values
print(len(datasetX))
# (9971, 33)

print(len(datasetY))


#按照窗口为1切分
windowssize = 1

trainbatchcount = len(datasetX) // windowssize
trainlen = len(datasetX) - windowssize + 1 #根据窗口调整后真正能训练的大小
X = np.zeros((trainlen,33 * windowssize)) #训练数据，33维特征
print(X.shape)
print(datasetY[0:6])
datasetY = datasetY[windowssize - 1:]


for i in range(trainlen):
    data = datasetX[i:(i+windowssize)]
    valuelist = ''
    for j,d in enumerate(data):
        valuelist = valuelist + '#' + d[0]
    valuelist = valuelist[1:]
    valuelist = valuelist.split('#')
    for k, value in enumerate(valuelist):
        X[i][k] = int(value)   #对应位置

# print(X[0:2])
# print(datasetX[0:4])
# print(datasetY[0:4])

print("-----------------------")
print(X.shape)
print(datasetY.shape)
print("-----------------------")


#active-->label
test_data_label_csv = pd.read_csv('milan-data-active-sensor-test.csv',usecols=[4])
#sensor-->input
test_data_input_csv = pd.read_csv('milan-data-active-sensor-test.csv',usecols=[5])
# print(data_csv)
# data_csv = data_csv.drop_duplicates()
# print(data_csv)
# print(data_csv.size)

#数据预处理
test_data_label_csv = test_data_label_csv.dropna() #去掉na数据
test_datasetY = test_data_label_csv.values
test_datasetY = test_datasetY.astype('int32') #整数标签
print(type(test_datasetY))
print(test_datasetY.shape)

test_data_input_csv = test_data_input_csv.dropna() #去掉na数据
test_datasetX = test_data_input_csv.values
print(len(test_datasetX))
print(len(test_datasetY))

testbatchcount = len(test_datasetX) // windowssize
testlen = len(test_datasetX) - windowssize + 1
test_X = np.zeros((testlen,33 * windowssize)) #训练数据，33维特征
print(test_X.shape)
test_datasetY = test_datasetY[windowssize - 1:]

for i in range(testlen):
    data = test_datasetX[i:(i+windowssize)]
    valuelist = ''
    for j,d in enumerate(data):
        valuelist = valuelist + '#' + d[0]
    valuelist = valuelist[1:]
    valuelist = valuelist.split('#')
    for k, value in enumerate(valuelist):
        test_X[i][k] = int(value)   #对应位置

#windows=1时 去除偏移部分，再预测与GRU网络比较
if windowssize==1:
   test_X=test_X[2:]
   test_datasetY=test_datasetY[2:]

print("-----------------------")
print(test_X.shape)
print(test_datasetY.shape)
print("-----------------------")


# 多项式朴素贝叶斯：sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# 主要用于离散特征分类，例如文本分类单词统计，以出现的次数作为特征值
# 参数说明：
# alpha：浮点型，可选项，默认1.0，添加拉普拉修/Lidstone平滑参数
# fit_prior：布尔型，可选项，默认True，表示是否学习先验概率，参数为False表示所有类标记具有相同的先验概率
# class_prior：类似数组，数组大小为(n_classes,)，默认None，类先验概率

start = time.perf_counter()
classifier = MultinomialNB()
classifier = classifier.fit(X,datasetY.ravel())
end = time.perf_counter()

print("Trained time:{:.6f} seconds".format(end - start))

pre_train=classifier.predict(X)
pre_test=classifier.predict(test_X) #测试集的预测标签

print("训练集accuracy：", accuracy_score(datasetY,pre_train) )
print("测试集accuracy：", accuracy_score(test_datasetY,pre_test) )


# #查看决策函数
# print('train_decision_function:\n',classifier.decision_function(X)) # (90,3)
print('predict_result:\n',classifier.predict(X))

# 生成性能报告,precision    recall  f1-score   support
print(classification_report(datasetY, pre_train))
print(classification_report(test_datasetY, pre_test, digits=6))

acdic={
    'Master_Bedroom_Activity':0,
    'Bed_to_Toilet':1,
    'Sleep':2,
    'Morning_Meds':3,
    'Watch_TV':4,
    'Kitchen_Activity':5,
    'Chores':6,
    'Leave_Home':7,
    'Read':8,
    'Guest_Bathroom':8,
    'Master_Bathroom':10,
    'Desk_Activity':11,
    'Eve_Meds':12,
    'Meditate':13,
    'Dining_Rm_Activity':14
}
classes = []
for k,v in acdic.items():
    classes.append(k)

matrix = confusion_matrix(test_datasetY, pre_test)
print(matrix)

"""classes: a list of class names"""
# Normalize by row
matrix = matrix.astype(np.float)
linesum = matrix.sum(1)
linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
matrix /= linesum
# 窗口大小*100像素
fig = plt.figure(figsize=(18, 10))
# 例如，“111”表示“1×1网格，第一子图”，
ax = fig.add_subplot(111)
cax = ax.matshow(matrix)
fig.colorbar(cax)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
for i in range(matrix.shape[0]):
    ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
ax.set_xticklabels([''] + classes, rotation=90)
ax.set_yticklabels([''] + classes)
#save
# plt.savefig(savname)
# 图片大小自适应
plt.tight_layout()
plt.show()

