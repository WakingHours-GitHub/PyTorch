import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
from model import *

# BATCH_SIZE = 1
LEARN_RATE = 0.01
EPOCH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 制作数据集
class PicDataset(Dataset):
    def __init__(self, train: bool = True):
        self.root_data = "./datas"
        self.read_data_path = os.path.join(self.root_data, "train" if bool else "test")
        self.pic_path = os.path.join(self.read_data_path, "picture_gray")
        self.pic_name_list = os.listdir(self.pic_path)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item) -> tuple:
        pic_names = self.pic_name_list[item]
        img = plt.imread(os.path.join(self.pic_path, pic_names))
        img = self.transform(img)
        label = pic_names[0:4]  # 分割标签
        label = self.one_hot(label)

        return img, label

    def __len__(self) -> int:
        return len(self.pic_name_list)

    def one_hot(self, one_label: str) -> torch.Tensor:
        return torch.tensor(np.eye(10)[[int(x) for x in one_label]], dtype=torch.float32)  # 直接相乘
        # return torch.ho

    def test(self) -> None:
        """
        测试Dataset是否可用, 并且返回的dataloader是否可用.
        :return:
        """
        print(self.pic_path)
        print(self[0])

        # print(self.one_hot(self[0][-1]))


def get_dataloader(istrain: bool = True):
    return DataLoader(
        dataset=PicDataset(istrain),  # 选择训练集和还是测试机
        batch_size=BATCH_SIZE,  # BATCH_SIZE来自model中
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    # 测试正常, 可以进行遍历


def calculate_accuracy(y_pred: torch.Tensor, label: torch.Tensor):
    """

    :param y_pred: [bz, 40]
    :param label: [bz, 4, 10]
    :return:
    """
    y_pred = torch.reshape(y_pred, shape=(-1, 4, 10))
    # print(torch.argmax(y_pred, dim=2))
    # print(torch.argmax(label, dim=2))
    acc = torch.all(
        torch.argmax(y_pred, dim=2) == torch.argmax(label, dim=2),
        dim=1
    ).float().mean()
    return acc


def train(isload: bool = True):
    # 准备数据集:
    dataloader = get_dataloader()

    # 准备模型:
    model = None
    load_model = "./model/SimpleModule.pth"
    if isload:
        if os.path.exists(load_model):
            model = torch.load(load_model)
        else:
            exit("未找到目录")
    else:
        model = SimpleModule()
    # 最后都需要进行to(device)
    model.to(device)
    model.train()  # 开启训练模式

    # 准备损失函数:
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.
    loss_fn.to(device)

    # 准备优化器
    SGD = torch.optim.SGD(params=model.parameters(), lr=LEARN_RATE)

    # 开始训练:
    for i in range(EPOCH):
        print("EPOCH:{}".format(i))
        for idx, (img, label) in enumerate(dataloader):
            # 转换数据
            img = img.to(device)
            label = label.to(device)

            # 使用定义的模型进行预测
            y_pred = model(img)
            acc = calculate_accuracy(y_pred, label)
            # print(label.shape)


            # y_pred = torch.reshape(y_pred, shape=(BATCH_SIZE, 4, 10)).float()
            # y_pred = torch.argmax(y_pred, dim=2).float()
            # label = torch.argmax(label, dim=2).float()

            # 使用loss_fn计算loss, loss是一组组合
            loss = loss_fn(y_pred, label)

            SGD.zero_grad()  # 清空梯度
            loss.requires_grad_(True)
            loss.backward()  # 反向传播, 计算梯度.
            SGD.step()  # update parameters

            print("index:{}, loss_value: {}, accuracy:{}".format(idx, loss.item(), acc))
            # break
        torch.save(model, load_model)


if __name__ == '__main__':
    # PicDataset().test()
    train()
