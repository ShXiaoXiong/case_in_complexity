import numpy #数组功能
import scipy.special #激活函数
import matplotlib.pyplot as plt #可视化
import random
import csv
import numpy #数组功能
import scipy.special #激活函数
import matplotlib.pyplot as plt #可视化
import random
import csv

class Robot():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #待传递节点数量
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #待传递学习率
        self.lr=learningrate 
        #设定初始连接权重：使用正态概率分布采样权重，也可以使用其他更为复杂的方法
        self.ihw=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.how=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #设定激活函数:使用sigmoid函数，一个常用的非线性激活函数，接受任何数值，输出0到1之间的某个值，但不包含0和1
        self.activation_function=lambda x:scipy.special.expit(x)

        #初始参数
        #位置（坐标法）
        self.x_coordinate=1
        self.y_coordinate=1
        #工作进度、成本和得分
        self.progress=0
        self.cost=0
        

        pass

    #观察获取输入：5个位置+进度+成本
    def check_event(self):
        inputs=[]#重建获取新的输入
        inputs.append((self.y_coordinate-1)*10+self.x_coordinate)#当前位置
        if self.x_coordinate<0 or self.y_coordinate<0 or self.x_coordinate>10 or self.y_coordinate>10:
            self.cost += 100 #圈外惩罚
        inputs.append(self.y_coordinate*10+self.x_coordinate)#上
        inputs.append((self.y_coordinate-1)*10+self.x_coordinate+1)#右
        inputs.append((self.y_coordinate-2)*10+self.x_coordinate)#下
        inputs.append((self.y_coordinate-1)*10+self.x_coordinate-1)#左
        inputs.append(self.progress)
        inputs.append(self.cost)

        return inputs
        #可能出现的最大值为120
      

    #行为函数：记录了工作进度和成本
    def action_1(self):
        self.x_coordinate +=1
        self.cost += 1
    def action_2(self):
        self.x_coordinate -=1
        self.cost += 1
    def action_3(self):
        self.y_coordinate +=1
        self.cost += 1
    def action_4(self):
        self.y_coordinate +=1
        self.cost += 1
    def action_5(self,board):#捡起
        self.cost += 1
        if board[(self.y_coordinate-1)*10+self.x_coordinate]==1:
            self.progress += 1
            
    def action(self,final_outputs,board):
        if numpy.argmax(final_outputs)==0:
            self.action_1()
        if numpy.argmax(final_outputs)==1:
            self.action_2()
        if numpy.argmax(final_outputs)==2:
            self.action_3()
        if numpy.argmax(final_outputs)==3:
            self.action_4()
        if numpy.argmax(final_outputs)==4:
            self.action_5(board)

    def query(self,inputs_list):
        #计算输出的过程 
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        return final_outputs#如果不写return，会返回一个None对象

board=numpy.zeros(100)
#生成10个不同的位置索引：random.sample选取列表中n个随机且独立的元素，返回一个列表
num = range(0, 100)   
nums = random.sample(num, 10)  
#将米放入棋盘中——机器人观察时，要获取索引位置的值即可
for record in nums:
    board[record]=1#board变为一个0和1组成的数组


lable=0#个体计数器
finanl_scores=[]#最终输出表格
     
n=Robot(7,100,5,0.2)
label = +1#要将label和均分合成一个列表
             
print(n.how)

while n.progress<10 and n.cost<200:#限定了两百步
    input_lists=n.check_event()#待处理数据
    input_lists=(numpy.asfarray(input_lists)/120.0 *0.99)+0.01#将输入值进行了预先处理
    inputs=numpy.array(input_lists,ndmin=2) #传递列表，转换为二维数组，不转置
    network_decision=n.query(inputs)
    n.action(network_decision,board)
    print('神经网络输出\n'+str(network_decision))
    print('行为编号'+str(numpy.argmax(network_decision)))

    print('坐标'+str(n.x_coordinate) + ',' +str(n.y_coordinate))
else:
    n.score=200-n.cost
    finanl_score=[[label,n.progress/10,n.score]]#个体标记、工作完成度、平均分




