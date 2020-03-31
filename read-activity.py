
import pandas as pd

# 15种行为
acdic={
    'Master_Bedroom_Activity':0,
    'Bed_to_Toilet':1,
    'Sleep':2,
    'Morning_Meds':3,
    'Watch_TV':4,
    'Kitchen_Activity':5,
    'Chores':6,           #nan 23
    'Leave_Home':7,
    'Read':8,
    'Guest_Bathroom':9,
    'Master_Bathroom':10,
    'Desk_Activity':11,
    'Eve_Meds':12,   #nan 19
    'Meditate':13,   #nan 17
    'Dining_Rm_Activity':14
}

sdic = {

 'D001':1,
 'D002':2,
 'D003':3,
 'M001':4,
 'M002':5,
 'M003':6,
 'M004':7,
 'M005':8,
 'M006':9,
 'M007':10,
 'M008':11,
 'M009':12,
 'M010':13,
 'M011':14,
 'M012':15,
 'M013':16,
 'M014':17,
 'M015':18,
 'M016':19,
 'M017':20,
 'M018':21,
 'M019':22,
 'M020':23,
 'M021':24,
 'M022':25,
 'M023':26,
 'M024':27,
 'M025':28,
 'M026':29,
 'M027':30,
 'M028':31,
 'T001':32,
 'T002':0

}

MAX = 30.5
MIN = 12.5

#传感器初值设置为-1
sensordata = [-1 for i  in  range(33)]

# total 455/2278   train:1823
TESTCOUNTLIMIT=[23,17,30,8,22,108,2,40,61,64,59,10,4,3,4]
testcount=[0 for i  in  range(15)]
testlist = []

# TRAINCOUNTLIMIT=[94,66,120,30,86,433,10,161,242,258,237,40,14,14,18]
# traincount=[0 for i  in  range(15)]
milanlist = []

i = 0
file_object = open('milan-data','r')
try: 
    for line in file_object:
        linedata  = line.strip().split('\t')
        # 活动编号
        aindex  = -1
        if(len(linedata)>=4 and len(linedata[3])>0):
            linedata[3] = linedata[3].strip()
            if (linedata[3][len(linedata[3])-5:]=='begin'):
                linedata[3]=linedata[3].replace(' begin','')
                aindex = acdic[linedata[3]]
                # print(aindex)
        
        # 异常数据处理
        if (linedata[2].strip()=='O'):
            linedata[2] = 'ON'
        if (linedata[2].strip()=='ON0'):
            linedata[2] = 'ON'
        if (linedata[2].strip()=='ON`'):
            linedata[2] = 'ON'

        # 传感器值映射
        sname = linedata[1].strip()
        sindex = sdic[sname]
        datalabel = linedata[2].strip()
        data = -1
        if datalabel in ['ON','OPEN']:
            data = 1
        if datalabel in ['OFF','CLOSE']:
            data = 0

        # 温度二值化
        if sindex in [0,32]:
            data = float(datalabel)
            data = (data - MIN) / (MAX - MIN)
            data = round(data,1)
            if data > 0.5 :
                data = 1
            else:
                data = 0

        # 更新传感器的值
        sensordata[sindex] = data

        # -1 表示传感器未全部初始化，从非-1点开始提取数据
        if sensordata.count(-1)==0 and aindex>-1 :
            # print(aindex)
            datatext = '#'.join(list(map(lambda x: str(x),sensordata)))
            if testcount[aindex]<TESTCOUNTLIMIT[aindex]:
                #测试数据集
                testlist.append([linedata[0].strip(),sname,datalabel,aindex,datatext])
                testcount[aindex] = testcount[aindex] + 1
            else:
                #训练数据集
                milanlist.append([linedata[0].strip(),sname,datalabel,aindex,datatext])

        i = i + 1
        # if i>=1000:
        #     break
except:
    print('error')
finally:
     file_object.close()

pf = pd.DataFrame(data=milanlist)
pf.to_csv("milan-data-active-sensor-train.csv")

pf = pd.DataFrame(data=testlist)
pf.to_csv("milan-data-active-sensor-test.csv")
