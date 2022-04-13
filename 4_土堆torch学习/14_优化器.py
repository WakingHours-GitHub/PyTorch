"""
所有的优化器都在:
torch.optim中

步骤:
    1. 构造一个优化器
        一般都是: optim.*(模型的参数(变量), 学习率, 以及一些模型特有的参数)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam([var1, var2], lr=0.0001)
        lr: 学习率.
    2. 调用step方法
    3.

例子:
for input, target in dataset:
    optimizer.zero_grad() # 清空梯度
    output = model(input) # 利用创建好的模型进行计算
    loss = loss_fn(output, target) # 计算loss值
    loss.backward() # 得到model中每个变量的一个梯度
    optimizer.step() # 根据上面得到的梯度, 对每一个变量进行更新

你应该还记得梯度下降算法, 是如何实现的,
就是根据之前函数值, 和学习率, 以及梯度, 进行更新当前的函数值,

"""


import torch
import torchvision
import torch.nn as nn
from torch import optim





def test01():
    # 设置我们的一个优化器:
    # 优化器对象 = optim.SGD() # 随机梯度下降

    # 然后开始这个循环:
    """
    伪代码:
    for (x, label) in batch:
        # 使用模型, 输入x, 得到y_pred,
        # y_pred与label进行比较, 计算loss
        # 清空梯度
        # loss.backward()反向传播, 计算梯度
        # 使用优化器, 对所有需要设置的变量的参数进行step(更新)
    每次循环一轮, 都是所有的batch数据被学习到了, 我们往往还需要在套用另外一个循环
    用来多次学习
    # 这个过程我们可以打断点, 看内部是如何发生的.
    
    """



if __name__ == '__main__':
    test01()