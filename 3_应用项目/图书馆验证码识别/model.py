import torch
import torch.nn as nn

# from main import BATCH_SIZE

BATCH_SIZE = 100


class VerificationCodeModule(nn.Module):
    def __init__(self):
        super(VerificationCodeModule, self).__init__()
        # 输入图像: [N, 1, 50, 130]
        self.dispose = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入通道
                out_channels=32,  # 输出通道, 多少个kernel
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2  # 填充区域
            ),
            # (50-5+4)/1 + 1 = 50
            # (130-5+4)/1 + 1 = 130
            # [N, 1, 50, 130]
            nn.ReLU(),
            nn.Conv2d(32, 46, (3, 3), (1, 1), 2),
            # x: [N, 46, 52, 132]
            nn.ReLU(),
            nn.MaxPool2d(2),  # 核多大, 步长就多大,
            # x: torch.Size([64, 46, 26, 66])
            nn.ReLU(),
            nn.Conv2d(46, 64, (3, 3), (1, 1), 1),
            nn.MaxPool2d(2),  # 核多大, 步长就多大,
            # torch.Size([64, 64, 13, 33])

        )

        self.flatten = nn.Flatten()

        self.full_connect = nn.Sequential(
            nn.Linear(64 * 13 * 33, 1024),
            nn.Linear(1024, 40)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.dispose(x)
        x = self.flatten(x)
        # torch.Size([64, 27456])
        x = self.full_connect(x)  # torch.Size([64, 40])
        x = torch.reshape(x, shape=(BATCH_SIZE, 4, 10))

        x = self.softmax(x)  # 一种概率映射
        x = self.flatten(x)

        return x


class SimpleModule(nn.Module):
    """
    仿照TenosrFlow中的网络模型在PyTorch中复现,
    """

    def __init__(self):
        super(SimpleModule, self).__init__()
        # 输入图像: [N, 1, 50, 130]
        self.layout1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.linear = nn.Linear(32 * 25 * 65
                                , 4 * 10)

    def forward(self, x):
        x = self.layout1(x)
        x = torch.reshape(x, shape=(-1, 32 * 25 * 65))
        x = self.linear(x)
        x = torch.nn.Softmax(dim=1)(x)
        return x


def model_test2():
    input = torch.zeros((64, 1, 50, 130))
    output = SimpleModule()(input)
    print(output.shape)


def model_test():
    input = torch.zeros((64, 1, 50, 130))
    output = VerificationCodeModule()(input)
    print(output.shape)


if __name__ == '__main__':
    # model_test()
    model_test2()
