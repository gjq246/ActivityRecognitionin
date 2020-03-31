import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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
name = []
for k,v in acdic.items():
    name.append(k)

y1 = [0.714286, 0.000000, 0.769231, 0.400000, 0.756757, 0.906542, 0.000000, 0.987654, 0.967742, 0.933333,0.844444,0.818182,0.000000,0.000000,0.000000]
y2 = [0.714286, 0.000000, 0.746667, 0.333333, 0.777778, 0.901786, 0.000000, 0.987342, 0.983607, 0.953846,0.832000,0.818182,0.000000,1.000000,0.333333]
y3 = [0.755556, 0.631579, 0.769231, 0.375000, 0.789474, 0.940092, 0.000000, 0.987654, 0.991736, 0.897638,0.909091,0.869565,0.400000,0.666667,0.000000]

#15
#15
#三个参数 起点为0，终点为3，步长为0.1
width = 0.2
xwidth = 0.9
x = np.arange(0,len(name)*xwidth,xwidth)


plt.bar(x, y1,  width=width, label='Naive Bayes')
plt.bar(x + width, y2, width=width, label='SVM', tick_label=name)
plt.bar(x + 2 * width, y3, width=width, label='GRU')

# 显示在图形上的值
# for a, b in zip(x,y1):
#     plt.text(a, b+0.01, round(b,2), ha='center', va='bottom')
# for a,b in zip(x,y2):
#     plt.text(a+width, b+0.01, round(b,2), ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.01, round(b,2), ha='center', va='bottom')

# plt.xticks()
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('F-Measure')
# plt.xlabel('line')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.xticks(rotation=45)
plt.tight_layout()
# plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
# plt.title("title")
plt.show()