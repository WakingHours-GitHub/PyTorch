import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional
import torchvision
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

# nn.Conv2d(
#
# )
# functional.conv2d(
#     input=,
#     weight=torch.normal(),
#     bias=torch.norm()
# )

# functional.conv2d(
#     input=, # 表示输入的Tensor, 并且输入Tensor的形状应该是: (minibatch, in_channels, iH, iw)
#     weight=, # 就是卷积核 -> (out_channels, in_channels, kH, kW)
#     bias= , # 就是偏置
#     stride=, #  就是在横向和竖向的步长, (sH, sW)
#     padding=, # 在图像的周围进行填充, 填充有多大. 默认不填充, 默认是为0.
#
# )
"""
torch.nn是将torch.nn.functional封装起来的, 是更高级的API
为了更加了解底层, 我们来看看torch.nn.functional中的Con2d是如何去做的吧


    下面我们来看看高级API是如何工作的:
    torch.nn.Conv2d(
        in_channels, # 输入的通道数, 单通道: 1, RGB通道: 3
        out_channels, # 输出的通道数, 实际上就是卷积核的个数
                # 理解: 就是不同的卷积核, 去看图片, 得到的结果也是不同的
                # -> 不同的视角,
        kernel_size, # 卷积核大小, 自动设置其中的数值, 满足二维高斯分布, 在训练过程中调整
        stride=1, # 步长
        padding=0, # 边缘填充
        dilation=1, # 卷积核对应距离
        groups=1, # 分组卷积
        bias=True, # 是否加上偏置
        padding_mode='zeros', #  填充方式
        device=None, 
        dtype=None
    )
    parameters:
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
        padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
"""


class MyConv2d(nn.Module):
    def __init__(self):
        super(MyConv2d, self).__init__()  # 注意, 在生成自己的module时, 需要调用super

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,  # 输出通道, 也就是多少个卷积核
            # 注意, 对于, size和stride是需要tuple, 所以就算是写一个数字也需要是使用元组
            kernel_size=(3, 3),  # (3, 3) 的一个卷积核
            stride=(1, 1),  # 步长为1
            padding=0,
            bias=True,

        )

    def forward(self, x):
        x = self.conv1(x)
        return x


def test02() -> None:
    """
    conv2d实战:
    可以看看VGG16网络结构
    :return:
    """
    dataset = datasets.CIFAR10(
        root="CIFAR10_data",
        train=False,  # 取出少量数据
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )

    with SummaryWriter("8_logs") as writer:
        for idx, (img, label) in enumerate(dataloader):
            print("img: ", img.shape)  # img:  torch.Size([64, 3, 32, 32])
            # 经过卷积层处理:
            img_conv2d = MyConv2d()(img)  # 返回的结果就是经过forward处理后的结果
            print("img_conv2d: ", img_conv2d.shape) # img_conv2d:  torch.Size([64, 6, 30, 30])
            # 可见, 经过卷积层后的图像大小
            writer.add_images("source_img", img, idx)
            img_conv2d = torch.reshape(img_conv2d, shape=(-1, 3, img_conv2d.size(2), img_conv2d.size(-1)))
            writer.add_images("conv2d_img", img_conv2d, idx)
            # 注意这里会报错, 因为conv2d处理完后的img_conv2d, channel为6, 不是3,
            # 所以我们这里做一个不严谨的操作: reshape以下, 这样img_conv2d就会被拉伸
            break  # 只看一轮就行

    # 注意这个父类, 重写了str()方法, 直接打印对时候, 就输出的是网络结构

    # 我们还可以打印这个模型: 这样我们就可以直观的看到网络结构:
    print(MyConv2d())
    """
    MyConv2d(
        (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
    )
    这样我们就可以很明显的看到我们的网络结构:
    
    """

    return None


def test01() -> None:
    """
    查看卷积层是如何工作的
    torch.nn.functional.conv2d()

    :return:
    """

    input = torch.tensor([
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ])

    # 定义卷积核:
    kernel = torch.tensor([
        [1, 2, 1],
        [0, 1, 0],
        [2, 1, 0]
    ])

    # 使用conv2D()API:
    # 显然, 我们的数据是二维的, 不符合functionalAPI的要求:

    print(input.shape)
    print(kernel.shape)

    input = torch.reshape(
        input=input,
        shape=(1, 1, 5, 5),  # batch_size=1, channel也为1
    )
    kernel = torch.reshape(
        input=kernel,
        shape=(1, 1, 3, 3)
    )

    output = functional.conv2d(
        input=input,
        weight=kernel,
        bias=None,
        stride=1,  # 横竖步长都为1
    )
    print(output)
    """
    tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
    """

    output2 = functional.conv2d(
        input=input,
        weight=kernel,
        bias=None,
        stride=2,  # 横竖步长都为2
    )
    print(output2)
    """
    tensor([[[[10, 12],
          [13,  3]]]])
    """

    # padding的作用, 就是在周围进行填充
    output3 = functional.conv2d(
        input=input,
        weight=kernel,
        bias=None,
        stride=2,
        padding=1
    )
    print(output3)

    # 卷积计算公式:
    # new_H = [(H+2*padding)/stride + 1]向下取整
    # 即N = (W-F+2P)/S + 1 然后向下取整

    return None


if __name__ == '__main__':
    # test01()
    test02()
