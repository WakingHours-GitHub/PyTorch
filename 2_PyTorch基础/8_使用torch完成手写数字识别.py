"""
思路:


"""
import torch
import numpy as np
import torchvision.datasets
from torchvision import transforms

def init():
    # 实现数据的就初始化
    mnist = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=True,
        transform=None
    )

def torch_transform():
    mnist = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=True,
        transform=None,
    )
    # 直接调用__call__
    img_tensor = transforms.ToTensor()(mnist[0][0]) # 看看转换回的结果
    # 那么这个ToTensor()就是可以帮助我们把image对象转换为我们的tensor对象

    print("img_tensor :", img_tensor)
    print("shape:", img_tensor.shape) # shape: torch.Size([1, 28, 28])
    # 这就是[C, H, W]



def my_transform():
    """
    模拟实现transform
    :return:
    """
    img = torch.tensor(
        data=np.random.randint(0, 255, size=12),
        dtype=torch.float32
    )
    # print(img) # tensor([ 40.,  96., 190., 134.,  82., 215., 126., 233.,  87., 229., 151., 190.])
    # 这就是动态图的好处, 那么直接定义, 直接运行

    img_reshape = img.view(2, 2, 3) # (H, W, C)
    print("img_reshape.shape:", img_reshape.shape) # img_reshape.shape: torch.Size([2, 2, 3])
    # print(img_reshape)
    # tensor([[[ 60.,   8., 216.],
    #          [236., 160., 160.]],
    #
    #         [[124., 185., 175.],
    #          [153., 139.,  18.]]])

    # img_reshape = torch.permute(img_reshape, [2, 0, 1])
    # print(img_reshape.shape) # torch.Size([3, 2, 2])
    print(img_reshape.permute(2, 0, 1).shape)


def test01():
    my_transform()


if __name__ == '__main__':
    # test01()
    torch_transform()