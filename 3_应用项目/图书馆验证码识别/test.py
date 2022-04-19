import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as F
import torch.nn as nn
import torch.nn.functional

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARN_RATE = 0.1
EPOCH = 5


# 准备数据集
class PictureDataset(Dataset):
    def __init__(self, train=True):
        super(PictureDataset, self).__init__()
        self.root_path = ".\\datas"
        self.data_path = os.path.join(os.path.join(self.root_path, "train" if train else "test", ), "picture_gray")
        self.pics_name = os.listdir(self.data_path)
        self.pics_path = [os.path.join(self.data_path, name) for name in self.pics_name]
        self.transform = F.Compose([
            F.Resize((50, 130)),
            F.ToTensor(),

        ])

    def test(self):
        # print(self.pics_name)
        print(self.pics_name[0])
        label = torch.tensor(label2onehot(self.pics_name[0][0:4]))

        print(label)

    def __getitem__(self, item):
        img_path = self.pics_path[item]
        # label = torch.tensor([label2onehot(x.split(".")[0]) for x in self.pics_name[item]], dtype=torch.float32)
        label = torch.tensor(label2onehot(self.pics_name[item][0:4]), dtype=torch.float32)
        img = Image.open(img_path)
        img = self.transform(img)  # 转换为Tensor对象.
        # print(label)

        return img, label

    def __len__(self):
        return len(self.pics_name)


class Dataset_use_cv(Dataset):
    def __init__(self, train=True):
        super(Dataset_use_cv, self).__init__()
        self.root = "./datas"
        self.data_path = os.path.join(self.root, "train" if train else "test")
        self.pic_gray_path = os.path.join(self.data_path, "picture_gray")
        self.pic_gray_names = os.listdir(self.pic_gray_path)

        # self.resize = F.Resize(
        #
        # )

    def get_dataloader(self) -> DataLoader:
        """
        已经测试好, 返回的就是一个DataLoader类型, 并且可用
        :return:
        """
        # print(self.pic_gray_names)

        # print(self.pic_gray_path)
        # print(self[0])
        return DataLoader(
            dataset=self,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2

        )

    def __getitem__(self, item):
        img_names = self.pic_gray_names[item]  # xxxx.png
        img = torch.tensor(np.array(plt.imread(os.path.join(self.pic_gray_path, img_names))))
        img = torch.reshape(img, shape=(-1, 50, 130))
        label = label2onehot(img_names[:4])

        return img, label

    def __len__(self):
        return len(self.pic_gray_names)


def label2onehot(one_label: str) -> torch.Tensor:
    """
    将标签快速转换为one_hot编码,

    https://blog.csdn.net/sinat_29957455/article/details/86552811
    :param one_label:
    :return:
    """
    """
    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    return torch.tensor(np.eye(10)[[int(i) for i in one_label]])


def show_img(self, item: int):
    img = cv.imread(self.pics_path[item])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 进行色彩空间的转换
    label = self.pics_name[item]

    plt.imshow(img)
    plt.title(label.split(".")[0])

    plt.show()


def get_dataloader() -> DataLoader:
    return DataLoader(
        dataset=PictureDataset(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True  # 丢弃最后一个batch
    )


def test_dataset() -> None:
    # print(PictureDataset().pics_path)
    print(PictureDataset()[0])
    print(PictureDataset()[1])
    # print(PictureDataset()[0][-1].shape)
    # PictureDataset().show_img(0)

    dataloader = get_dataloader()
    for idx, (img, label) in enumerate(dataloader):
        print(img.shape)
        print(label)
        y_pred = simplemodule(img)
        print(y_pred.shape)

    return None


# 构建模型
VFCM = VerificationCodeModule()
VFCM.to(device=device)
# VFCM = torch.load("./model/VFCM.pth")

simplemodule = SimpleModule()
# simplemodule = torch.load("./model/SimpleModule.pth")
# print(simplemodule)
simplemodule.to(device=device)
#
# 损失函数
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.MultiLabelSoftMarginLoss()
loss_fn.to(device=device)  # 切换设备

# 优化器对象
Adam = torch.optim.Adam(params=simplemodule.parameters(), lr=LEARN_RATE)


def new_train():
    dataloader = Dataset_use_cv().get_dataloader()
    for i in range(EPOCH):
        print("begin {} epoch train".format(i))

        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)  # torch.Size([64, 1, 50, 130])
            label = label.to(device)  # [-1, 4, 10]
            # print(img)
            # print(label)

            y_pred = simplemodule(img)  # [-1, 40]
            y_pred = torch.reshape(y_pred, shape=(-1, 4, 10))

            # accuracy = (torch.nn.functional.one_hot(torch.argmax(y_pred, dim=2)).float() == label.float()).float().sum()

            accuracy = torch.all(torch.argmax(y_pred, dim=2) == torch.argmax(label, dim=2),
                                 dim=1).float().mean()  # 横向看, 是否相等
            # print(accuracy)

            y_pred = torch.reshape(y_pred, shape=(-1, 40))
            label = torch.reshape(label, shape=(-1, 40)).float()
            # print(y_pred)
            # print(label)
            # break
            loss = loss_fn(y_pred, label) # 计算loss

            Adam.zero_grad() # 清空梯度
            loss.backward() # 反向传播, 计算梯度.
            Adam.step() # 更新

            print(loss.item(), accuracy.item())
            if idx % 100 == 0:
                torch.save(simplemodule, "./model/SimpleModule.pth")
    torch.save(simplemodule, "./model/SimpleModule.pth")


# 开始训练
def train():
    """
    开始
    :return:
    """
    dataloader = get_dataloader()
    for i in range(EPOCH):

        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            label = nn.Flatten()(label)
            # print(img.shape, label.shape)
            # 计算: y_pred
            y_pred = VFCM(img)  #
            # print(y_pred.shape) # torch.Size([64, 40])

            # print(label.shape) # torch.Size([64, 40])
            loss = loss_fn(y_pred, label)

            # 计算损失率
            y_pred = torch.reshape(y_pred, shape=(BATCH_SIZE, 4, 10))
            label = torch.reshape(label, shape=(BATCH_SIZE, 4, 10))
            # print(y_pred)
            # print(label)
            # break
            accuracy = (y_pred == label).float().mean()
            print("idx: {}, loss: {}, acc:{}".format(idx, loss.item(), accuracy))
            if idx % 100 == 0:
                torch.save(VFCM, "./model/VFCM.pth")

            # 更新
            Adam.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播, 计算梯度
            Adam.step()  # 更新参数.
            # break
        # break


if __name__ == '__main__':
    new_train()
    # print(label2onehot("0001"))
    # train()
    # test_dataset()
    # PictureDataset().test()
