import os.path
import matplotlib.pyplot as plt
import random
import numpy as np
import pywin32_bootstrap
from  torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn


device = torch.device("cuda") # captain GPU device

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # torch.Size([64, 1, 28, 28])
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), 1)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*14*14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = torch.reshape(x,shape=(-1, 32*14*14))
        x = self.linear(x)

        return x

    def test(self):
        input = torch.ones(size=(1, 1, 28, 28))
        print(self(input).shape)



def train(isload:bool=False):
    dataset = torchvision.datasets.MNIST(
        root=r"D:\PyCharm\PyTorch\2_PyTorch基础\mnist_data",
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    if isload:
        if not os.path.exists("./mnist_model/mnist.pth"):
            exit("没有该目录")

    else:
        m = model().to(device)


    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    optim = torch.optim.Adam(params=m.parameters(), lr=0.01)



    for idx, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        # print(img.shape) # torch.Size([64, 1, 28, 28])
        y_pred = m(img)

        loss = loss_fn(y_pred, label)

        optim.zero_grad() # 清空梯度

        loss.backward()
        optim.step()

        # 计算准确率
        accuracy = (torch.argmax(y_pred, dim=1)==label).float().mean()

        print(loss.item(), accuracy)

        if not( idx % 100):
            if not os.path.exists("./mnist_model"): # 如果没有该文件夹,
                os.makedirs("./mnist_model")
            torch.save(m, "./mnist_model/mnist.pth")



def test_model_effort():
    """

    测试模型小姑
    :return:
    """
    dataset = torchvision.datasets.MNIST(
        root=r"D:\PyCharm\PyTorch\2_PyTorch基础\mnist_data",
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    )
    test_img_list = random.sample([data[0] for data in dataset], 6)
    plt.figure(1, figsize=(8, 8), dpi=100)

    model = torch.load("./mnist_model/mnist.pth")
    model = model.to(device)

    for idx, test_img in enumerate(test_img_list):
        plt.subplot(2, 3, idx+1)
        test_img_ndarray = np.array(test_img)
        # (1, 28, 28)
        test_img_ndarray = np.reshape(test_img_ndarray, newshape=(28, 28, 1))

        plt.imshow(test_img_ndarray, cmap=plt.cm.gray) # 显示灰色图

        # 使用模型进行预测:
        test_img = test_img.to(device)
        test_img = test_img.view((1,1,28,28))
        # print(test_img.shape)
        y_pred = model(test_img)
        # print(y_pred.shape)
        result = torch.argmax(y_pred, dim=1)

        plt.title("OCR:" + str(result.item()))  # torch.Size([1, 1, 28, 28])

    plt.show()





if __name__ == '__main__':
    # train()

    test_model_effort()
    # model().test()