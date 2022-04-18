"""
最大池化的作用:
最大池化也被称之为下采样:

torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

input – input tensor (\text{minibatch} , \text{in\_channels} , iH , iW)(minibatch,in_channels,iH,iW), minibatch dim optional.
kernel_size – size of the pooling region. Can be a single number or a tuple (kH, kW)
    就是卷积核.
stride – stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size
    就是步长, 默认是与卷积核大小相同,
padding – Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
    填充
dilation – The stride between elements within a sliding window, must be > 0.
    dilation: 就是差距, 也就是进行卷积操作时, 是需要进行乘积的, 那么dilation就表示原图像卷积时候的距离 -> 空洞卷积
ceil_mode – If True, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.
    是否保留? 就是向上取整, 还是向下取整. 当越过边界时
return_indices – If True, will return the argmax along with the max values. Useful for torch.nn.functional.max_unpool2d later

那么池化到底是做什么的?
    池化就是提取特征的, 通过设置核的大小, 提取窗口内最明显的(maxPool就是提取最大值)
    本质上就是降采样, 可以大幅度减少网络的参数量

input: (N, C, H, W)
output: (N, C, H, W)

"""
import torch
import torchvision.datasets
import torchvision.transforms as F
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch.nn as nn


class MyMaxPool(nn.Module):
    def __init__(self):
        super(MyMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size = 3, # 传入一个int, 默认就是H, W都等于该int
            ceil_mode=True,
        )
    def forward(self, x):
        return self.pool(x)


def test02():
    dataset = torchvision.datasets.CIFAR10(
        root="CIFAR10_data",
        train=False,
        transform=F.ToTensor(),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )
    with SummaryWriter("./9_logs") as writer:
        for idx, (img, label) in enumerate(dataloader):
            # 处理图片:
            img_pool = MyMaxPool()(img)
            writer.add_images("img", img, idx)
            writer.add_images("img_pool"
                              "", img_pool, idx)



def test01():
    input = torch.tensor([
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ], dtype=torch.float32)
    input = torch.reshape(input, shape=(-1, 1, 5, 5))
    print(input.shape)  # torch.Size([1, 1, 5, 5])
    output = MyMaxPool()(input)
    print("output", output)
    print("output.shape", output.shape)  # output.shape torch.Size([1, 1, 2, 2])



if __name__ == '__main__':
    # test01()
    test02()







