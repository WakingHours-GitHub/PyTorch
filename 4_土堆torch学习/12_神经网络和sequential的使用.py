"""


sequential: 序列:
使用起来也很简单, 就是有一个简单的网络模型实现:

torch.nn.Sequential(*args)

model = nn.Sequential(
    nn.Conv2d(),
    # 放入网络结构
    # ...
)
out = model(x)


然后我们使用Sequential, 来搭建一个我们最小的网络模型

简单实现一下CIFAR10的经典网络模型



"""

import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as F
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import *



class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 输入图像:
        # x: [batch_size, 3, 32, 32]
        self.conv1=nn.Conv2d(
            in_channels=3, # 输入的通道数
            out_channels=32, # 输出的通道数
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, ),
        )
        # 输出: x[batch_size, 32, 32, 32]
        # 卷积层输出公式:
        # H_out = (H-F+2P)/s + 1
        #

        # x: [batch_size, 32, 32, 32]
        self.maxpool1=nn.MaxPool2d(
            kernel_size=2
        )
        # x: [batch_size, 32, 16, 16]
        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, )
        )
        # x: [batch_size, 32, 16, 16]
        self.maxpool2=nn.MaxPool2d(
            kernel_size=2,
        )
        # x: [batch_size, 32, 8, 8]

        self.conv3=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, ),
            padding=(2, )
        )
        # x[batch_size, 64, 8, 8]
        self.maxpool3=nn.MaxPool2d(
            kernel_size=2
        )
        # x: [batch_size, 64, 4, 4]

        # 目前为止图像是: x: [batch_size, 64, 4, 4]
        self.flatten = nn.Flatten() # 和torch.flotten()一样
        # 只对dim=1(包括)以后的维度进行平坦.

        # 到这里时, 输入图像已经是:
        # 线性层也称之为全连接层.
        # 定义的就是一个矩阵的行和列, in就是行, out就是列
        # 也就是这样的一个矩阵: [in, out] # 然后这一层要左的就是矩阵相乘
        # 1024 = 64*4*4
        self.linear1=nn.Linear(
            in_features=64*4*4,
            out_features=64
        )
        # 经过线性层,
        # 为64
        self.linear2=nn.Linear(
            in_features=64,
            out_features=10 # 最后分成10个类别.
        )

    # 在构建网络的时候, 我们需要时时刻刻的关注我们x形状的一个变化
        self.model1 = torch.nn.Sequential(

        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# 使用Sequential创建模型:
class SequentialModule(nn.Module):
    def __init__(self):
        super(SequentialModule, self).__init__() # 继承子类
        # Sequential就是一个序贯模型
        # 当模型较为简单时, 我们可以使用Sequential类来实现简单模型的顺序连接
        # 调用也很简单, 就是直接传入一个x, 然后他会从上到下, 依次执行.

        self.sequential = nn.Sequential(
            # conv2d:(in, out, kernel, stride, padding)
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=(1, ), padding=(2, )),
            # 注意, 要么写整数, 要么写一整个元组, 不要省略
            # # maxpool2d: 常用就只有个kernel_size, 然后步长与kernel_size一样.
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), (1, ), 2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), (1, ), 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10) #
            # 这样我们的序贯模型就定义完了. # 但是其中, 我们一定注意我们输入的形状变化.
        )

    def forward(self, x):
        # 直接返回就可以了, 非常简单, x会经过我们的序贯模型最终得到输出
        return self.sequential(x)

def test03():
    """
    使用连续序贯模型, 进行简单的测试.
    我们定义好各个层的网络, 然后模型就会自动按照顺序执行我们的模型,forward

    :return:
    """
    s_module = SequentialModule()
    print(s_module)
    """
    SequentialModule(
      (sequential): Sequential(
        (0): Conv2d(3, 32, kernel_size=(5,), stride=(1,), padding=(2, 2))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): Conv2d(32, 32, kernel_size=(5,), stride=(1,), padding=(2, 2))
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(32, 64, kernel_size=(5,), stride=(1,), padding=(2,))
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Flatten(start_dim=1, end_dim=-1)
        (7): Linear(in_features=1024, out_features=64, bias=True)
        (8): Linear(in_features=64, out_features=10, bias=True)
      )
    )

    """
    input = torch.ones(size=(64, 3, 32, 32), dtype=torch.float32)
    print(input.shape) # torch.Size([64, 3, 32, 32])
    output = s_module(input)
    print(output.shape) # torch.Size([64, 10])




def test02():
    # 当网络建立完成之后, 我们就可以来看一下我们的网络结构了
    # 注意, 打印出实例化对象
    print(MyModule())
    """
    MyModule(
      (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1,), padding=(2,))
      (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1,), padding=(2,))
      (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(32, 64, kernel_size=(5,), stride=(1,), padding=(2,))
      (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear1): Linear(in_features=1024, out_features=64, bias=True)
      (linear2): Linear(in_features=64, out_features=10, bias=True)
    )
    """
    dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=F.ToTensor(),

    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,

    )

    for idx, (img, label) in enumerate(dataloader):
        print(img.shape)
        img = MyModule()(img) # 调用模型, 直接调用forward及逆行处理
        print(img.shape)

        break

def test01():
    """
    进行一个简单的实验
    :return:
    """
    # 进行一个简单的检验

    input = torch.ones(size=(64, 3, 32, 32)) # 初始化变量
    output = MyModule()(input)
    print(output.size()) # torch.Size([64, 10]) # 说明正确


    # 我们也可以使用tensorboard进行 网络模型的 可视化    
    with SummaryWriter("12_logs") as writer:
        writer.add_graph( # 增加计算图. 
            model=MyModule(), # 放入模型 # 将
            input_to_model=input, # 模型的输入变量
        )




if __name__ == '__main__':
    # test02()
    # test01()
    test03()