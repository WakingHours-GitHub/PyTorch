import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision  # 导入数据
import matplotlib.pyplot as plt

"""
步骤:
    识别手写图片步骤： 
    Load data
    Build Model
    Train
    Test
    
"""


# 实战开始:

########################################################################################################################
# 辅助代码
# 画出曲线, 画出loss的图像
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc="upper right")
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# 可视化图片
def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# one_hot编码函数
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)

    return out


########################################################################################################################

batch_size = 512
# 加载数据集:
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 转换为Tensor
                                   torchvision.transforms.Normalize(  # 正则化
                                       (0.1307,), (0.3081,))]  # 一次处理多张图片.
                               )),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 转换为Tensor
                                   torchvision.transforms.Normalize(  # 正则化
                                       (0.1307,), (0.3081,))]  # 一次处理多张图片.
                               )),
    batch_size=batch_size, shuffle=False
)

# 直观感受:
# 拿到数据
x, y = next(iter(train_loader))
print(x.shape, y.shape)  # 查看形状
# torch.Size([512, 1, 28, 28]) torch.Size([512])
plot_image(x, y, "image sample")


# 2. 创建网络: 三层非线性层:
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 三层网络:
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)  # 最终一定是由分类类别所决定给的.

    # 前向传播
    def forward(self, x):
        # x: [b, 1, 28, 28]
        x = F.relu(self.fc1(x))
        # H2 = relu(h1W2 + b2)
        x = F.relu(self.fc2(x))
        # 因为最终问题是一个多分类问题, 所以我们往往使用softmax,进行
        # 或者直接使用, 均方差作为最后的loss计算
        # 这里我们没有使用softmax, 而是直接使用输出值作为我们的结果
        x = self.fc3(x)

        return x


# 初始化网络
net = Net()
# parameters就是表示我们要更新的权重
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练:
for epoch in range(3):  # 将数据集遍历三次

    # 对batch迭代
    for batch_idx, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)

        # x: [b, 1, 28, 28] y:[512]
        # 使用net之前, 我们还需要flat操作
        # [b, 1, 28, 28] -> [b, 784]
        x = x.view(x.size(0), 28 * 28)
        # [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)

        # loss = mse(x, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        # 如何优化这个loss呢, 我们先需要一个损失
        #
        optimizer.zero_grad()  # 梯度清零
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()  # 迭代梯度

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 最后我们应该得到了比较好的一组权重
# loss只是帮助我们优化的, 通过找到loss的梯度, 来优化权值
# 实际上我们更关注的是准确度
# 4. 准确度的测试

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("acc:",acc) # 输出准确率

# 打印结果值
