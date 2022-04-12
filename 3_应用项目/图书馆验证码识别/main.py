import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


# 准备数据集
class PictureDataset(Dataset):
    def __init__(self, train=True):
        self.root_path = ".\\datas"
        self.data_path = os.path.join(os.path.join(self.root_path, "train" if train else "test", ), "picture_gray")
        self.pics_name = os.listdir(self.data_path)
        self.pics_path = [os.path.join(self.data_path, name) for name in self.pics_name]
        self.transform = F.Compose([
            F.Resize((50, 130)),
            F.ToTensor(),

        ])

    def __getitem__(self, item):
        img_path = self.pics_path[item]
        label = [x.split(".")[0] for x in self.pics_name]
        img = Image.open(img_path)
        img = self.transform(img)
        print(label)

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
    # print(PictureDataset()[0:2])
    # PictureDataset().show_img(0)
    dataloader = get_dataloader()
    # for idx, (img, label) in enumerate(dataloader):
    #     print(img.shape)
    #     print(label)
    #     break

    return None


# 构建模型
# 损失函数
# 优化器对象
# 开始训练


def main():
    """
    开始
    :return:
    """


if __name__ == '__main__':
    # main()
    test_dataset()
