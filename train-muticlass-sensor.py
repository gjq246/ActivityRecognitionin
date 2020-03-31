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


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


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
print(len(datasetY))

data_input_csv = data_input_csv.dropna() #去掉na数据
datasetX = data_input_csv.values
print(len(datasetX))
print(len(datasetY))
print(len(datasetX))

#按照窗口为7切分
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
            for k, value in enumerate(valuelist):
                hot[j][k] = int(value)   #对应位置
        data.append((hot,datay[i + look_back - 1][0]))
    return data

train = create_dataset(datasetX,datasetY,windowssize)

# print(train[0])
# print(train[1])
# print(train[2])

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
print(len(train))  #5958
batch_size = 1  #时序网络batch_size为1比较好，但是训练比较慢
input_dim = 33  #33个传感器
hidden_dim = 128  # 128 
output_dim = 15 #行为
n_layers = 2
learn_rate = 0.0003 #0.0003 #学习率

model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
model.to(device)

criterion = nn.CrossEntropyLoss() #多分类
optimizer = torch.optim.RMSprop(model.parameters(), lr=learn_rate)

model.train()
epoch_times = []
EPOCHS = 2000  

hist = np.zeros(EPOCHS) # 用来记录每一个epoch的误差

# Start training loop
for epoch in range(1,EPOCHS+1):
    start_time = time.perf_counter()
    h = model.init_hidden(batch_size) #初始化隐藏节点的值
    avg_loss = 0.
    counter = 0
    correct  = 0
    batchcount = len(train) // batch_size
    xstart = 0
    xend = 0
    for i in range(batchcount):
        xend = xstart + batch_size
        counter += 1
        h = h.data #上一次隐藏节点的值，记忆功能
        model.zero_grad()
        traindata = train[xstart:xend]
        outputy = np.empty(shape=(0))
        xbatch = []
        for x, label in traindata:
            # outputy是自己用的可以自己定义维度，答案        
            outputy = np.append(outputy, [label])
            xbatch.append(x)
        xbatch = torch.Tensor(xbatch)
        intputx = xbatch.to(device).float()
        outputy = torch.from_numpy(outputy).to(device).long()
        out, h = model(intputx, h)
        # torch.Size([1, 2])   1个结果，输出维度2，最大的值的下标为答案
        # 计算准确率
        for j in range(out.size(0)):
            _, predicted = out[j].max(0)
            if train[xstart+j][1] == predicted:
                correct = correct + 1

        xstart = xend
        loss = criterion(out, outputy)  #[batchsize,outputdim],[batchsize]
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        #内部一次
        # if counter%500 == 0:
        #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {:.6f}".format(epoch, counter, len(train), avg_loss/counter))
    # 所有样本一轮
    current_time = time.perf_counter()
    accuracy  = (correct / (batchcount * batch_size)) * 100
    hist[epoch - 1] = avg_loss/len(train)
    if epoch % 2 == 0:
        print("-" * 50)
        print("Epoch {}/{} Done, Total Loss: {:.8f}".format(epoch, EPOCHS, avg_loss/len(train)))
        usedtime = sum(epoch_times[:epoch])
        print("Time Elapsed for Epoch: {:.2f} seconds".format(usedtime))
        remaintime = ((usedtime / epoch) * EPOCHS - usedtime)/(60 * 60)
        print("Time Remained for Epoch: {:.8f} hours".format(remaintime))
        print("Accuracy: {:.2f}%".format(accuracy))
    epoch_times.append(current_time-start_time)
    if epoch % 500 == 0:
        print("Save Model......")
        torch.save(model, 'gru-mutilclass-sensor-w3-'+str(epoch)+'.pt')
        print("Save Loss......")
        pf = pd.DataFrame(data=hist)
        pf.to_csv('gru-loss-sensor-w3-'+str(epoch)+'.csv')
# 多轮训练
print("=" * 50)
print("Window Size:{},Batch Size:{}".format(windowssize,batch_size))
print("Total Training Time: {:.8f} hours".format(sum(epoch_times)/(60*60)))
torch.save(model, 'gru-mutilclass-sensor-w3.pt')

# # 打印Loss
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.plot(hist, label="Training loss")
ax.legend()
plt.show()

