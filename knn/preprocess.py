#conding:utf-8

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv("./dating.txt")
print(data)

#数据归一化 是为了除去量纲
#1.1实例化一个转化器
transfer = MinMaxScaler(feature_range=(0,1))
#1.2调用fit transfrom方法
minmax_data = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
print("经过归一化处理之后的数据为:\n",minmax_data)
#归一化 鲁棒性很差 最大值和最小值容易受到异常点的影响，只适合传统精确小数据场景

#数据标准化
#异常点影响比较小
#2.1实例化一个转换器  x-mean/方差
transfer = StandardScaler()
standed_data = transfer.fit_transform(data[["milage","Liters","Consumtime"]])
print("经过标准化处理之后的数据为:\n",standed_data)
