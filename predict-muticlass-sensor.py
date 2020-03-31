import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
from sklearn.model_selection import train_test_split

from matplotlib.ticker import MultipleLocator
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


#  将数据归一化到0~1之间，80%作为训练集，其余作为测试集
#active-->label
data_label_csv = pd.read_csv('milan-data-active-sensor-test.csv',usecols=[4])
#sensor-->input
data_input_csv = pd.read_csv('milan-data-active-sensor-test.csv',usecols=[5])
# print(data_csv)
# data_csv = data_csv.drop_duplicates()
# print(data_csv)
# print(data_csv.size)

#数据预处理
data_label_csv = data_label_csv.dropna() #去掉na数据
datasetY = data_label_csv.values
datasetY = datasetY.astype('int32') #整数标签
print(len(datasetY))

data_input_csv = data_input_csv.dropna() #去掉na数据
datasetX = data_input_csv.values
print(len(datasetX))
print(len(datasetY))
print(len(datasetX))
#按照窗口为3切分
windowssize = 3
#0,1,2,3,4,5,6,7: len=8   window=3  8-3=5,  4+3=7
#1,0,1   ,2 ,3,4,5
# 输入数据和标签全部规整成一行，按照窗口大小切片成对
def create_dataset(datax,datay,look_back=2):
    data=[]
    for i in range(len(datax) - look_back + 1):
        awindow=datax[i:(i + look_back)]
        # print(datay[i + look_back - 1][0])
        hot = [ [0 for m in range(33)] for n in range(windowssize)]  #33个传感器
        for j, sensorvalue in enumerate(awindow):           
            valuelist = sensorvalue[0].split('#')
            # print(valuelist)
            for k, value in enumerate(valuelist):
                hot[j][k] = int(value)   #对应位置
        data.append((hot,datay[i + look_back - 1][0]))
    return data
test = create_dataset(datasetX,datasetY,windowssize)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  #多分类用softmax激活函数
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.softmax(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

# Setting common hyperparameters
print("Test Total:",len(test))  #5958，442
batch_size = 1  #大于1时，要与训练一致效果最好

model = torch.load('gru-mutilclass-sensor-w3-2000.pt')
model.to(device)
model.eval()

counter = 0
correct  = 0
batchcount = len(test) // batch_size

xstart = 0
xend = 0

# 初始化隐藏节点值
h = model.init_hidden(batch_size)

#预测结果保存
pre_output = np.zeros(batchcount * batch_size)
test_label = np.zeros(batchcount * batch_size)

for i in range(batchcount):
    
    xend = xstart + batch_size
    counter += 1
    # 上一次的隐藏节点值
    h = h.data
    testdata = test[xstart:xend]

    outputy = np.empty(shape=(0))
    xbatch = []
    for x, label in testdata:
        # outputy是自己用的可以自己定义维度，答案        
        outputy = np.append(outputy, [label])
        xbatch.append(x)
    xbatch = torch.Tensor(xbatch)
    intputx = xbatch.to(device).float()
    outputy = torch.from_numpy(outputy).to(device).long()
    out, h = model(intputx, h)

    # 计算准确率
    for j in range(out.size(0)):
        _, predicted = out[j].max(0)
        pre_output[xstart+j] = predicted #保存预测值
        test_label[xstart+j] = test[xstart+j][1] #分离标签值
        if test[xstart+j][1] == predicted:
            correct = correct + 1
    xstart = xend
print("correct:",correct)
print("predict total:",batchcount * batch_size)
accuracy  = (correct / (batchcount * batch_size)) * 100
print("Accuracy: {:.2f}%".format(accuracy))
print("测试集accuracy：", accuracy_score(test_label,pre_output) )
print(classification_report(test_label, pre_output, digits=6))
matrix = confusion_matrix(test_label, pre_output)
print(matrix)

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
# https://matplotlib.org/gallery/color/colormap_reference.html
# https://blog.csdn.net/liuchengzimozigreat/article/details/90477501
cax = ax.matshow(matrix,cmap=plt.cm.GnBu_r)
fig.colorbar(cax)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
for i in range(matrix.shape[0]):
    ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
ax.set_xticklabels([''] + classes, rotation=45)
ax.set_yticklabels([''] + classes)
#save
# plt.savefig(savname)
# 图片大小自适应
plt.tight_layout()
plt.show()
