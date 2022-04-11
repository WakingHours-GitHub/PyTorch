# 使用pytorch完成手写数字的识别
import os.path
import numpy as np
import torchvision  # 导入torch中的, 与视觉相关的包
import torch
from torch.utils.data import DataLoader  # 导入DataLoader
from torchvision import transforms  # 导入, 对非Tensor对象, 做处理的函数
import torch.nn.functional as F  # 导入激活函数.
import torch.nn as nn

BATCH_SIZE = 100
TEST_BATCH_SIZE = 200

# 1. 准备数据集
def get_MNIST_dataloader(train=True, barch_size=BATCH_SIZE): # 默认是训练集
    # 定义transforms方法. 使用Compose将转换方法进行组合
    transforms_f = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的形状和通道数相同.
        # 因为是针对每个通道计算出来的mean和std
    ])
    # 获取mnist的datasets
    mnist_dataset = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=train,
        transform=transforms_f,
        download=False
    )
    # 然后将datasets, 放到批处理队列, 也就是dataloader中去
    mnist_dataloader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,  # 选择dataset
        batch_size=barch_size,  # 批处理队列大小
        shuffle=True,  # 是否shuffle, 是否打乱

    )

    return mnist_dataloader


# 2. 构建模型
class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(1 * 28 * 28, 28, bias=True)  #

        self.fc2 = nn.Linear(28, 10)

        pass

    def __call__(self, input):
        """
        一个全连接层
        一个激活层
        一个全连接层

        :param input: [batch, 1, 28, 28]
        :return:
        """
        # 因为要输入到全连接层, 就是要进行矩阵相乘, 所以我们需要将input转换为二阶张量
        # 也就是矩阵, 所以使用view方法, 形状的转换
        # 1. 修改形状, 使其满足矩阵相乘
        ## input.view([input.size(0), 1*28*28])
        x = input.view([-1, 1 * 28 * 28])
        # 因为torch是动态图, 运行到这的时候, 已经知道input的具体tensor变量了, 所以这里可以使用.size()获取batch_size的值
        # 但是TF1版本中不可以, 因为是静态图, 在TF中我们可以使用-1, 自动计算.
        # 我们也不可以直接传入batch_size, 因为如果最后batch_size样本量可能不够, 所以batch_size不确定, 不过我们可以令drop_last=True, 丢弃最后一个batch_size
        # 所以我们不能直接将batch_size固定起来, 我们更多的还是使用-1
        #

        # 2. 进行全连接操作
        x = self.fc1(x)  # 这里直接传入刚才定义的这个fc1对象, 直接调用__call__()方法
        # 3. 激活函数 (经过激活函数, tensor形状没有发生变化)
        x = F.relu(x)  #

        x = self.fc2(x)

        return F.log_softmax(x, dim=-1) # 指定在那个dim计算sorfmax, 实际上是对每个行, 也就是最后一个维度


model = MnistModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if os.path.isdir("./MNIST_model/model"): # 增强健壮性.
    if os.path.isfile("./MNIST_model/model/model.pt") and os.path.isfile("./MNIST_model/model/optimizer.pt"):
        model.load_state_dict(torch.load("./MNIST_model/model/model.pt"))
        optimizer.load_state_dict(torch.load("./MNIST_model/model/optimizer.pt"))
        print("加载成功")
    else:
        print("加载失败")

def train(epoch):
    """
    实现训练的过程
    :param epoch:
    :return:
    """
    data_loader = get_MNIST_dataloader(train=True)
    for idx, (x, label) in enumerate(data_loader): # 训练所有的样本
        optimizer.zero_grad() # 将梯度置为0
        y_pred = model(x) # 调用模型, 得到预测值
        loss = F.nll_loss(y_pred, label) # 计算损失函数
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        # 计算准确率:

        # accuracy =
        # print(y_pred.size()) # torch.Size([100, 10]) # 就是[barch_size, one-hot编码]
        pred = y_pred.data.max(1, keepdim=True)[-1] # 获取最大值问题
        accuracy = torch.eq(pred, label.data.view_as(pred)).sum()/BATCH_SIZE

        # break

        if not(idx % 100):
            print("idx",idx, "loss:", loss.item())
            print(accuracy)
            # 模型保存
            torch.save(model.state_dict(), "./MNIST_model/model/model.pt")
            torch.save(optimizer.state_dict(), "./MNIST_model/model/optimizer.pt")



def test():
    # 加载数据集
    test_dataloader = get_MNIST_dataloader(train=False, barch_size=TEST_BATCH_SIZE) # 获取测试数据加载器
    loss_list = []
    acc_list = []

    # 遍历数据集合;
    for idx, (x, y_true) in enumerate(test_dataloader):
        # 接下来我们就可以进行预测了:
        with torch.no_grad(): # 不会追踪梯度
            y_pred = model(x) # 通过模型得到, 预测值
            # 这里, y_pred: [barch_size, 10]
            # 而 y_true: [barch_size]
            # 所以我们应该对y_pred找到最大概率值所在的哪个位置, 然后与y_true进行比较
            # 得到bool矩阵(逻辑矩阵), 然后对每一个batch进行求和, 除以batch_size, 这样就得到了一组batch的准确率.
            # 那么我们如何获取最大值所在的位置嗯?: 就是使用 .max(dim=, keepdim=)
            # 返回值是一个元组, 0，是tensor, 1是所在的位置.
            # 所以准确率的计算:
            acc = y_pred.max(dim=-1)[-1].eq(y_true).float().mean()
            loss = F.nll_loss(y_pred, y_true)
            # print(y_pred.shape, y_true.shape)
            loss_list.append(loss.item())
            acc_list.append(acc.item())
    # 计算平均损失
    loss_mean = np.mean(loss_list)
    acc_mean = np.mean(acc_list)
    print(f"本次, 测试集(batch): {TEST_BATCH_SIZE}, 平均损失: {loss_mean}, 平均准确率: {acc_mean}")




if __name__ == '__main__':
    for i in range(3): # 训练整个三次
        # train(i)
        pass
    test()