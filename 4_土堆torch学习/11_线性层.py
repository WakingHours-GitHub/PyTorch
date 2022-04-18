"""
Normalization
正则化层. 可以优化网络.

Recurrent Layers
一些写好的网络结构,

Transformer Layers
也是一种写好的网络结构

Linear Layers
线性层

Dropout Layers
    nn.Dropout()
    就是在训练的过程中, 随机丢弃一些其中的因素,
    按照p(概率)去丢失. 为了防止过拟合现象
Distance Layers
    就是计算误差
Loss Function:
    就是损失函数.


这里主要看线性层:
    其实也就是全连接层 FULL_
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    torch封装的非常好, 只要输入, 和输出以及是否要偏置就可以实现
实际上就是矩阵相乘, 使得最后的结果得到我们想要的形状
因为是矩阵相乘, 所以输入的tensor必须是一个二维的
所以一般再使用Linear之前, 我们一般会使用reshape和view
所以一般是: [barch, ...]然后与矩阵相乘, 根据我们的要求,  得到最终的结果
input:(*, input)
output: (*, output)


in_features – size of each input sample
out_features – size of each output sample
bias – If set to False, the layer will not learn an additive bias. Default: True


torchvision.models
这里面提供了一些非常经典对于图像方面的网络结构.
别人已经训练好的模型, 你拿过来直接使用就好了.

 """

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as F

BATCH_SIZE = 64

class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(
            in_features=3*32*32,
            out_features=10
        )
    def forward(self, x):
        return self.linear(x)


def test01():
    # 加载数据集
    dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=F.ToTensor(),

    )
    # 加载数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,

    )

    for idx, (img, label) in enumerate(dataloader):
        print(img.shape)
        # 再介绍一个函数:
        # img = torch.flatten(img) # 直接摊平.
        # print(img.shape)
        # torch.flatten()直接摊平，
        img = torch.reshape(img, (-1, 3*32*32))
        img_out = MyLinear()(img)
        print(img_out.shape)
        # torch.Size([64, 10])
        # 当摊平后:
        # torch.Size([10])
        break





if __name__ == '__main__':
    test01()