"""
一个我们非常常用的激活函数:
F.relu()

Input: (*)(∗), where *∗ means any number of dimensions.
Output: (*)(∗), same shape as the input.


非线性激活层, 就是给我们的图像增加非线性激活能力.


"""
import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as F



# 我们自己的非线性激活函数模型:
class MyNon_linear(nn.Module):
    def __init__(self):
        # self.to()
        super(MyNon_linear, self).__init__()
        self.relu = nn.ReLU(
            inplace=False, # 是否原地修改
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.relu(x)

def test02():
    dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=F.ToTensor(),
        download=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )
    with SummaryWriter("10_logs") as writer:
        for idx, (img, label) in enumerate(dataloader):
            img_out = MyNon_linear()(img)
            writer.add_images("img", img, idx)
            writer.add_images("img_relu", img_out, idx)



def test01():
    input = torch.tensor([
        [1, -0.5],
        [-1, 3]
    ])
    print(input)
    """
    tensor([[ 1.0000, -0.5000],
        [-1.0000,  3.0000]])
    """
    out_put = MyNon_linear()(input) #

    print(out_put)
    """
    tensor([[1., 0.],
        [0., 3.]])
    # 可见, 经过Relu处理后, 改变了, <0的都为0, >0的为本身
    """

if __name__ == '__main__':
    # test01()
    test02()