"""

损失函数
作用:
1. 衡量我们输出和目标之间的一个差距
2. 用于反向传播(为我们的更新提供一定依据)

# 简单的来看一下:
　

我们往往都是根据我们的任务来选择loss function
我们只需要关心, 我们使用的这个输入和输出的形状是什么样子就可以了




"""

import torch
import torch.nn as nn

# 先来定义一下我们的输入和label案例吧:
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
labels = torch.tensor([1, 2, 5], dtype=torch.float32)

# 转换成四维: 也就是带batch的高阶张量
inputs = inputs.view((1, 1, 1, 3))# torch.Size([1, 1, 1, 3])
labels = labels.view(1, 1, 1, 3)

def test04():
    """
    自动计算梯度

    :return:
    """

def test03():
    """
    交叉熵损失。
    交叉熵损失常常用到多分类问题中.
    公式就是: -log(y_pred.*label)/sum(log(y_pred))
    label应该是一个one_hot编码, 但是, 这里只需要传入label真实值对应的位置就可以了
    input: (C), (N, C), (N, C, ...) 计算K维度的loss
    target: (), (N), (N, ...)
    # 注意, 输入一般是(barch_size, 多分类的类别)
    # 而target只有(batch_size) 然后每个数, 对应的就是真实的类别, (的编码)
    # 返回的是scalar就是一个标量张量.
    :return:
    """


def test02():
    """
    计算MSELoss, 记住, 是MSE, MeanSquaredError
    :return:
    """
    result = nn.MSELoss()(inputs, labels)
    print(result) #tensor(1.3333)

    # 直接计算误差平方和, 常常在回归(连续)问题中用到MSE
    # 误差平方和, 就是差值的平方, 然后除以个数, 得到均方误差





def test01():
    """
    nn.L1Loss(reduction="sum")(pred, label)
    # 这就计算pred和label之间的差值的绝对值
    :return:
    """
    result_avg = nn.L1Loss()(inputs, labels)
    print(result_avg) # tensor(0.6667)

    result_sum = nn.L1Loss(reduction="sum")(inputs, labels)
    print(result_sum)

    # tensor(0.6667)
    # tensor(2.)
    # L1Loss就是直接计算对应位置之差的绝对值.



if __name__ == '__main__':
    # test01()
    test02()