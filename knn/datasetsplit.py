# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split



def plot_iris(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue="target", fit_reg=False)
    plt.title("------")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1.数据集获取
    # 1.1小数据集获取
    iris = load_iris()
    # 数据可视化，把数据用DataFrame存储
    iris_d = pd.DataFrame(iris['data'], columns=["Seoal_len", "Seoal_width", "Petal_len", "Petal_width"])
    iris_d["target"] = iris.target
    #print(iris_d)
    #plot_iris(iris_d, "Seoal_width", "Seoal_len")
    #4.数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
   # print("训练集的特征值是:\n",x_train)
   # print("测试集的特征值是:\n", x_test)
    #print("训练集的目标值是:\n",y_train)
   # print("测试集的目标值是:\n", y_test)
    #print("训练集的目标值形状:\n",y_train.shape)
    #print("测试集的目标值形状:\n", y_test.shape)
    #4，1
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
