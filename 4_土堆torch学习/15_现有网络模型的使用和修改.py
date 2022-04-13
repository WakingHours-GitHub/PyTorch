"""
我们学习一些经典的网络模型, 并且修改其中的一些参数

这次我们来介绍一个对图片继续分类的常用模型
VGG:
torchvision
def vgg16(
    pretrained: bool = False,  # 表示是否加载数据集imgNet和已经训练好的模型参数
    progress: bool = True # 是否加载下载进度条.
    ):

很多框架都会将VGG16作为一个前置的网络结构, 一般用VGG16提取一些感兴趣的特征
然后在加上自己的网络结构, 输出自己想要的结果.


"""

import torchvision
import torchvision.transforms as F
import torch.nn as nn



def test03():
    """
    将现有的网络结构修改
    :return:
    """
    vgg16_params = torchvision.models.vgg16(
        pretrained=True,
        progress=True
    )
    print(vgg16_params) # 先查看网络结构
    # 然后我们需要修改classifier中的最后一层
    vgg16_params.classifier[6] = nn.Linear(4096, 10)
    print(vgg16_params)
    #     (6): Linear(in_features=4096, out_features=10, bias=True)
    # 修改成功.
    # 这样子我们就修改成功了



def test02():
    """
    在VGG16的基础上再添加一层Linear, 使其输出我们自己的网络结构
    :return:
    """
    vgg16 = torchvision.models.vgg16(
        pretrained=False
    )
    vgg16.add_module(
        name="add_linear",
        module=nn.Linear(1000, 10)
    )
    print(vgg16)
    """
    # 这样就成功添加到我们的后面了
        (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    (add_linear): Linear(in_features=1000, out_features=10, bias=True)
    """

    # 那如果我们想添加到classifier之后:
    vgg16.classifier.add_module("linear", nn.Linear(1000, 10))
    print(vgg16)
    """
    # 这样就成功添加到classifier当中.
      (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    (linear): Linear(in_features=1000, out_features=10, bias=True)
  )
    """


def test01():
    # train_dataset = torchvision.datasets.ImageNet(
    #     root="./imageNet_data",
    #     split="train",
    #     download=True,
    #     transform=F.ToTensor()
    # )
    # RuntimeError: The dataset is no longer publicly accessible. You need to download the archives externally and place them in the root directory.
    # 可以看到, 该数据模型已经无法使用了
    # 这个imgNet数据集足足有100多gb, 所以这个训练集不适合讲解

    vgg16 = torchvision.models.vgg16(
        pretrained=False,
        # 如果为False就只加载这个模型类, 返回对象
        # 如果为True, 那么就下载关于imgNet的模型参数, 需要下载
    )
    # 可以利用debug的方式, 来查看该对象内部具体属性
    # 也就可以看到其中的具体变量的值
    #
    print(vgg16) # 打印网络架构
    # 最后输出的是1000, 也就是说他要分1000中类型
    # 我们来看一下imageNet这个数据集, 它包括十几万张图片, 然后有1000类的图片

    # (6): Linear(in_features=4096, out_features=1000, bias=True)
    # 那么如何将该模型应用到我们自己的模型当中去呢
    # 例如我们现在想要将该模型应用到CIFAR10这个数据中去, 他的输出是10个类别
    # 可以看到他最后的结果是输出1000个类别, 那么我们可以将这个1000修改为10
    # 或者可以在加入一个Linear层, 输入是1000, 输出是10





if __name__ == '__main__':
    # test01()
    # test02()
    test03()
