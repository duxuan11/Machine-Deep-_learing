# conding:utf-8

# 1.获取数据集
# 2.数据基本处理
# 3.特征工程
# 4.机器学习(模型训练)
# 5.模型评估

from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#1.获取数据集
iris = load_iris()
#2.数据基本处理
#2.数据分割
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22,test_size=0.2)

#3.特征工程
#3.1实例化一个转换器
transfer = StandardScaler()
#3.2调用fit_transform方法
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
#4.机器学习(模型训练)
#4.1实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5)
#4.2模型训练
estimator.fit(x_train,y_train) #训练集上的特征值，目标值进行训练

#5.模型评估
#5.1输出预测值
y_pre = estimator.predict(x_test) #对测试集的特征值进行预测
print("预测值是:\n",y_pre)
print("预测值与真实值对比:\n",y_pre==y_test) #检测预测值与测试集真实目标值
#5.2输出准确率 score
ret = estimator.score(x_test,y_test)  #测试测试集的特征值是否与其真实目标值对应
print("准确率是:\n",ret)
