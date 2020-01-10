#棋盘：10*10

#米的数量：10，随机分布不重复

#机器人初始位置：（0,0）
#机器人可做的事：观察前后左右一个格子，无成本
#机器人行为2：移动一格，成本函数（计数器）
#机器人行为3：拾取米，成本函数（计数器）
#问题：是否存在某种模式的路线？

#机器人神经网络（遗传算法）：输入：输出：4种移动或拾取米
#机器人得分：再大燃料量（200）-总成本
#遍历方法得分：再大燃料量（200）-(10*10+10+10)=80
#如果得分能超过80，则存在某种更合理的线路

#设定一个连接权重，按照该权重完成游戏，记录得分
#根据得分，选择高分进行遗传
#重复上述过程，直到得到唯一一个survivor


#训练
#查询功能

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
        #工作进度、成本
        self.progress=0
        self.cost=0

        pass

    #观察获取输入：5个位置+进度+成本###这个问题应该使用决策树才更合理
    def check_event(self):
        inputs=[]#获取新的输入
        
        
        inputs.append((self.y_coordinate-1)*10+self.x_coordinate)#当前位置的索引值
        if self.x_coordinate<0 or self.y_coordinate<0 or self.x_coordinate>10 or self.y_coordinate>10:
            self.cost += 100 #高额圈外惩罚
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

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #完全相同的计算，因此在循环中要重写
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数


        targets=numpy.array(targets_list,ndmin=2).T#传递列表，转换为二维数组
        
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=numpy.dot(self.how.T,output_errors)#点乘
        #反向传递，更新how权重
        self.how += self.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        self.ihw += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))#点乘
        pass


#构建神经网络实例
#输入：需要考虑的问题是：观察前后左右的情况、工作进度、成本
#输出：根据输出的值大小，选择5种行为中的其中一种

#2^3=8
#每代淘汰一半，3代最后剩下一个survior。各代中的每个个体训练100张同样的地图，总训练量(2^3+2^2+2^1)*100=1400张地图

label=0#个体计数器
finanl_scores=[]#最终输出表格
epoch=2
for e in range(epoch):#世代之中完成交配

    ###设置棋盘，并将棋盘存入列表中
    boards=[]
    for xx in range(10):#多少张图，每张图中完成游戏，棋盘不动
        #创建棋盘数组
        board=numpy.zeros(100)
        #生成10个不同的位置索引：random.sample选取列表中n个随机且独立的元素，返回一个列表
        num = range(0, 100)   
        nums = random.sample(num, 10)  
        #将米放入棋盘中——机器人观察时，要获取索引位置的值即可
        for record in nums:
            board[record]=1#board变为一个0和1组成的数组
        boards.append(board)
     
    for record in range(2):#多少个个体
        n=Robot(7,100,5,0.2)
        label += 1#要将label和均分合成一个列表
        
        scores=[]
        progresses=[]
     
        for board in boards:
            while n.progress<10 and n.cost<200:#完成工作，或规定燃料用完
                input_lists=n.check_event()#待处理数据
                input_lists=(numpy.asfarray(input_lists)/120.0 *0.99)+0.01#将输入值进行了预先处理
                inputs=numpy.array(input_lists,ndmin=2) #传递列表，转换为二维数组，不转置
                network_decision=n.query(inputs)
                n.action(network_decision,board)
            else:
                n.score=200-n.cost
                scores.append(n.score)#输出得分列表
                progresses.append(n.progress/10.0)
                print(n.score)
                
        finanl_score=[[label,numpy.mean(progresses),numpy.mean(scores)]]#个体标记、工作完成度、平均分，两百张图
        finanl_scores.append(finanl_score)





        

