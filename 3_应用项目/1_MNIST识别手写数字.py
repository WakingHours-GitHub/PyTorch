import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import *
from torch.utils.data import DataLoader  # 注意, 是大写,
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

# 使用GPU进行训练.
device = torch.device("cuda:0")


# 构建数据集加载器集合

def get_mnist_dataloader(istrain=True, batch_size=BATCH_SIZE):
    transform_F = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # 获取dataset
    mnist_dataset = torchvision.datasets.MNIST(
        root="../2_PyTorch基础/mnist_data",  # 读取文件
        train=istrain,  # 是否训练
        transform=transform_F,  # 需要对数据集进行的操作
        download=False
    )

    # 放到batch队列中去
    mnist_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 有效核心数
    )

    return mnist_dataloader


# 构建模型:
class MnistModel(nn.Module):
    def __init__(self, isload=True):
        super(MnistModel, self).__init__()

        # 初始化优化器
        self.optimizer = None

        # 构架模型:
        # input(batch, 1, 28, 28)
        # 需要先将
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入的channels数
            out_channels=32,  # 输入的channels数
            kernel_size=(3, 3),  # 卷积核形状
            stride=(1, 1),  # 步长
            padding=1,  # 边缘填充, (是围绕图片填充一圈
            padding_mode="zeros",
            bias=True  # 添加偏置
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3, 3),  # 池化卷积层大小
            stride=2,  # 步长为2, 说明减少一圈
            padding=0,
        )
        self.full = nn.Linear(
            in_features=32 * 13 * 13,
            out_features=10
        )

        # self.full = nn.Linear()

    def forward(self, x):
        """
        输入: [batch_size, 1, 28, 28]
        :param x:
        :return:
        """
        x = self.conv1(x)
        # x: [batch_size, 1, 28, 28]
        x = F.relu(x)  # 激活函数不改变形状
        # print(x.shape)

        x = self.maxpool(x)  # 池化
        # x: [batch_size, 32, 13, 13]
        x = F.relu(x)
        # print(x.shape)
        # 进行全连接层:
        # 首先我们需要将x形状, 变为二阶张量. 然后进行矩阵相乘也就是全连接层
        x = x.view((-1, 32 * 13 * 13))
        # 进行全连接层:
        x = self.full(x)
        x = F.relu(x)
        # print(x.shape) # 最后得到的x就是: [batch_size, 10]
        # print(F.log_softmax(x)) # 这是经过softmax函数后的结果. 也就是各个类别的概率值
        # 返回
        return F.log_softmax(x, dim=-1)  # 注意默认是以最后一个维度进行计算, 也就是计算一列, 得到行向量

    def train_model(self, isload=True):
        # 这里的self就指代model本身
        # 定义优化器
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        if isload:  # 是否训练
            if os.path.isdir("./model/MNIST/"):  # 增强健壮性.
                if os.path.isfile("./model/MNIST/model.pt") and os.path.isfile(
                        "./model/MNIST/optimizer.pt"):
                    self.load_state_dict(torch.load("./model/MNIST/model.pt"))
                    self.optimizer.load_state_dict(torch.load("./model/MNIST/optimizer.pt"))
                    print("load model success")
                else:
                    print("load fail")
        else:  # 不虚拟蓝
            print("not load model")

        batch_data = get_mnist_dataloader()
        for index, (x, y_true) in enumerate(batch_data):
            x = x.to(device)
            y_true = y_true.to(device)

            self.optimizer.zero_grad()  # 重置梯度
            y_pred = self(x)
            loss = F.nll_loss(y_pred, y_true)  # 计算损失
            loss.backward()  # 反向传播, 就是计算梯度
            self.optimizer.step()  # 更新参数
            acc = torch.eq(y_pred.data.max(-1)[-1], y_true).float().mean()
            print("index:{}, loss: {}, acc: {}".format(index, loss.item(), acc.item()))
            # break

            # 保存模型
            if not (index % 100):
                torch.save(self.state_dict(), "./model/MNIST/model.pt")
                torch.save(self.optimizer.state_dict(), "./model/MNIST/optimizer.pt")

            # self(x)
            # break
        # self.forward()

    def test(self):
        pass


if __name__ == '__main__':
    model = MnistModel()

    for i in range(10):
        model.train_model(isload=True)
