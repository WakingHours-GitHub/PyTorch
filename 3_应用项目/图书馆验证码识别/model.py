import torch
import torch.nn as nn

class VerificationCodeModule(nn.Module):
    def __init__(self):
        super(VerificationCodeModule, self).__init__()
        # 输入图像: [N, 1, 50, 130]
        self.dispose = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入通道
                out_channels=32, # 输出通道, 多少个kernel
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2 # 填充区域
            ),
            # (50-5+4)/1 + 1 = 50
            # (130-5+4)/1 + 1 = 130
            # [N, 1, 50, 130]
            nn.ReLU(),
            nn.Conv2d(32, 46, (3, 3), (1, 1), 2),
            # x: [N, 46, 52, 132]
            nn.ReLU(),
            nn.MaxPool2d(2), # 核多大, 步长就多大,
            # x: torch.Size([64, 46, 26, 66])
            nn.ReLU(),
            nn.Conv2d(46, 64, (3, 3), (1, 1), 1),
            nn.MaxPool2d(2),  # 核多大, 步长就多大,
            # torch.Size([64, 64, 13, 33])

        )

        self.flatten = nn.Flatten()

        self.full_connect = nn.Sequential(
            nn.Linear(64*13*33, 1024),
            nn.Linear(1024, 40)
        )

    def forward(self, x):
        x = self.dispose(x)
        x = self.flatten(x)
        # torch.Size([64, 27456])
        x = self.full_connect(x) # torch.Size([64, 40])

        return x



def model_test():
    input = torch.zeros((64, 1, 50, 130))
    output = VerificationCodeModule()(input)
    print(output.shape)
    
if __name__ == '__main__':
    model_test()
