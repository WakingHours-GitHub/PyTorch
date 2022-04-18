



"""
网络搭建:
神经网络: Neural NetWork

TORCH.NN
These are the basic building blocks for graphs:
类别:

    Containers: 容器, 或者说骨架, 我们就是要根据
    Convolution Layers: 卷积层
    Pooling layers: 池化层
    Padding Layers: padding
    Non-linear Activations (weighted sum, nonlinearity): 非线性激活
    Non-linear Activations (other) #
    Normalization Layers:
    Recurrent Layers
    Transformer Layers
    Linear Layers
    Dropout Layers
    Sparse Layers
    Distance Functions
    Loss Functions
    Vision Layers
    Shuffle Layers
    DataParallel Layers (multi-GPU, distributed)
    Utilities
    Quantized Functions
    Lazy Modules Initialization

"""
from typing import Callable

from torch.utils.hooks import RemovableHandle

"""
Containers: 
Module          Base class for all neural network modules. -> 非常重要的
Sequential      A sequential container.
ModuleList      Holds submodules in a list.
ModuleDict      Holds submodules in a dictionary.
ParameterList   Holds parameters in a list.
ParameterDict   Holds parameters in a dictionary.


Module:
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module): # 必须继承父类nn.Module
    def __init__(self):
        super().__init__() # 使用父类的初始化方法, 进行初始化
        # 定义自己的网络模型架构: 
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x): # 进行前向传播
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
"""

import torch.nn as nn
import torch.nn.functional as F # 导入激活函数方法


# 创建我们自己的模型:
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()


    # 注意, 这个函数是我们必须要实现的.
    # 并且__call__()方法里, 调用的就是该方法, 所以我们可以直接使用对象(), 当成函数来去使用
    def forward(self, x): # 这个就很简单, 只是对输入进行+1
        x = x + 1
        return x


if __name__ == '__main__':
    print(MyModule()(1))