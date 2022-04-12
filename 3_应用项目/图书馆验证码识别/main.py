import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as F
import torch.nn as nn

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARN_RATE = 0.01
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
        label = torch.tensor(label2onehot(self.pics_name[0][0:4]), dtype=torch.float32)
        img = Image.open(img_path)
        img = self.transform(img)
        # print(label)

        return img, label


    # def __getitem__(self, item):
    #     img_path = self.pics_path[item]
    #     label = self.pics_name[item]
    #     # print(label)
    #
    #     # img = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) for path in img_path]
    #     img = np.array([np.reshape(plt.imread(path), (1, 50, 130)) for path in img_path])
    #     # label = [x.split(".")[0] for x in label]
    #     # print(label)
    #     img = torch.tensor(img)
    #     # print(img.shape)
    #     return img, label

    def __len__(self):
        return len(self.pics_name)




def label2onehot(one_label:str) -> torch.Tensor:
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
    return np.eye(10)[[int(i) for i in one_label]]



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
    # print(PictureDataset()[0][-1].shape)
    # PictureDataset().show_img(0)

    dataloader = get_dataloader()
    for idx, (img, label) in enumerate(dataloader):
        print(img.shape)
        print(label)
        break

    return None


# 构建模型
VFCM = VerificationCodeModule()
VFCM.to(device=device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device=device)

# 优化器对象
Adam = torch.optim.Adam(params=VFCM.parameters(), lr=LEARN_RATE)

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
            y_pred = VFCM(img)
            # print(y_pred.shape) # torch.Size([64, 40])

            # print(label.shape) # torch.Size([64, 40])
            loss = loss_fn(y_pred, label)


            # 计算损失率
            y_pred = torch.reshape(y_pred, shape=(BATCH_SIZE, 4, 10))
            label = torch.reshape(label, shape=(BATCH_SIZE, 4, 10))
            accuracy = (y_pred == label).float().mean()
            print("idx: {}, loss: {}, acc:{}".format(idx, loss.item(), accuracy))



            # 更新
            Adam.zero_grad() # 清空梯度
            loss.backward() # 反向传播, 计算梯度
            Adam.step() # 更新参数.
            # break
        # break


if __name__ == '__main__':
    # print(label2onehot("0001"))
    train()
    # test_dataset()
    # PictureDataset().test()