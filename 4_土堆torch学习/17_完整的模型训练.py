"""

通过CIFAR10, 来去完成我们的一个完整的模型训练套路
这个数据集是比较简单的. 并且是容易上手的


一个完整的模型搭建顺序:
    加载数据集dataset, 装载成dataloader
    搭建网络模型, 并给定输入进行测试
    定义损失函数, 根据我们的问题, 定义合适的损失函数
    定义优化器对象, 设置训练的模型参数, 并且设置学习率
    设置训练网络的一些参数, 例如, 训练次数, 训练轮数
    开始训练, 我们往往要训练多轮, 然后每一轮是放入一组batch_size
    加入tensorboard, 进行可视化
        - 网络结构的可视化
        - loss的变化
        - 准确率(accuracy)
        - 每次测试集的图片
    保存模型: torch.save

每一轮训练之后, 我们可以看看在测试集上的效果, 用来评估模型的训练情况
注意, 在测试集上, 我们不需要调优了, 也就是我们无需追踪梯度
API: with torch.no_grad(): # 无法追踪梯度的空间.
    # 代码


# 注意, loss是什么? 显然, loss无法直观的证明结果的有效性, 只能说明模型在更新,
对于一个分类问题, 准确率, 往往是更加重要的.
那么如何计算准确率呢?:
    首先我们要知道, 给定一个输入, 经过输出, 得到是多个类别的一组概率值
    那么去argmax()就可以求出来最大值所在的位置
    或者使用.max(), 返回的就是最大值以及最大值所在的位置, .max()[1]返回的就是最大值的位置
    我们可以设置dim=来指定维度.
    然后与label进行比较, 返回一个逻辑矩阵, 然后.float32, 转换为float类型
    然后使用.sum()或者.mean()计算求和或者是平均数.

我们还经常看到, 在模型训练之前
我们会加一句, model.train()
    设置模型进入训练状态:
    对一部分网络层有作用, 例如: Dropout,  BatchNorm, etc
在模型评估之前, 我们会加上一个model.eval()
    也是对一些特定的网络层起作用,
如果你的网络结构中有这些层, 那么你可以在训练开始之前使用xxx.train(), 然后再训练
再评估模型之前使用xxx.eval(), 然后再进行你的网络测试

"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os, os.path

# 搭建神经网络:
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
# 但是, 注意, Flatten不影响batch的大小。, 也就是Flatten层, 不压缩, batch_size的大小, 只压缩后面的大小,
# 这一步我们往往也使用view来去操作, 因为全连接层需要矩阵的乘法, 所以我们需要将高维数据转换为矩阵(二维)

# 我们往往将所创建好的神经网络单独放到一个文件当中, from xxx import *
# 直接引入.

# 定义GPU设备:
devise = torch.device("cuda:0")


def accuracy_test():
    output = torch.tensor([
        [0.1, 0.2],
        [0.05, 0.4],
    ])
    print(output.argmax(dim=0))  # 当为0时, 我们就竖着看, 谁最大, 并且返回一个列向量.

    print(output.argmax(dim=1))  # 当为1时, 我们就横向看, 谁最大, 并且返回一个列向量.
    preds = output.argmax(dim=1)
    target = torch.tensor([1, 1])
    print(preds == target)  # 返回的就是一个逻辑矩阵 # tensor([True, True])
    # print((preds == target).float().sum()) # tensor(2.) # 这就直接计算出来sum值了, 我们可以再除以Batch_size求得平均值
    # 或者我们可以直接.mean()
    print((preds == target).float().mean())  # 直接求取平均值 # tensor(1.)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 创建网络模型
        # 输入x: [batch_size, 3, 32, 32]
        self.sequential = nn.Sequential(
            # x: [batch_size, 3, 32, 32]
            nn.Conv2d(3, 32, (5, 5), (1, 1), 2),
            # x:[bt, 32, 32, 32]
            nn.MaxPool2d(2),
            # 核为多少, 步长就自动为多少
            # x: [bt, 32, 16, 16]
            nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            # x: [bt, 32, 16, 16]
            nn.MaxPool2d(2),
            # x: [bt, 32, 8, 8]
            nn.Conv2d(32, 64, (5, 5), (1, 1), 2),
            # x: [bt, 64, 8, 8]
            nn.MaxPool2d(2),
            # x: [bt, 64, 4, 4]
            # 进行flatten摊平操作
            # 注意, Flatten层, 不对batch_size进行修改
            nn.Flatten(),
            # x: [bt, 64*4*4]
            nn.Linear(64 * 4 * 4, 64),
            # x: [bt, 64]
            # 因为最后是分类问题, 所以我们还需要经过全连接层输出我们最终类别的概率值
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.sequential(x)


def test_for_model():
    """
    对于模型的测试
    :return:
    """
    input = torch.ones(size=(64, 3, 32, 32))
    output = MyModel()(input)
    print(output.shape)  # torch.Size([64, 10])
    # 最终得到的结果正确.

    pass


def test01() -> None:
    if not os.path.isdir("./17_logs"):
        os.makedirs("./17_logs")
    writer = SummaryWriter("17_logs")
    BATCH_SIZE = 64
    # 准备数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=True,
        transform=F.ToTensor(),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=F.ToTensor(),
    )
    # 打印一下长度:
    print("len(train_dataset): ", len(train_dataset))  # len(train_dataset):  50000
    print("len(test_dataset): ", len(test_dataset))  # len(test_dataset):  10000

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,  # 删除最后一组batch
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    # 查看图片形状
    # print(train_dataset[0][0].shape) # torch.Size([3, 32, 32])

    # 定义模型对象
    model = MyModel()
    model.to(devise)  # 将模型也加载到cuda上

    input_to_model = torch.ones(size=(BATCH_SIZE, 3, 32, 32))
    # 模型可视化.
    writer.add_graph(model=model, input_to_model=input_to_model)

    # 定义损失函数:
    # 因为是多分类问题, 所以使用交叉熵损失函数来作为损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(devise)

    # 定义优化器对象
    learn_rate = 0.01
    # 定义优化器的时候, 我们需要指定需要训练模型当中哪些网络参数, 并且还需要设置学习率
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate)

    # 开始训练:
    # 定义训练参数:
    epoch = 10  # 训练十论

    for i in range(epoch):
        print(" -->> current epoch: {} <<-- ".format(i))
        # 有时候我们会看到有的代码再训练之前会加上:
        model.train()
        # 这是表示模型开始训练了, 如果你的模型有一些特殊的层, 那么这是我们开启模型的训练模型就可以了
        for step, (img, target) in enumerate(train_dataloader):
            # print(img.shape) # torch.Size([64, 3, 32, 32])
            # 使用GPU进行训练
            # 将CPU Tensor对象, 转变为GPU-Tensor
            img = img.to(devise)
            target = target.to(devise)
            # 模型也to -> cuda:0上面

            y_pred = model(img)  # 经过神经网络计算, 得到预测值
            loss = loss_fn(y_pred, target)  # 计算loss值, 这步骤传入的参数与刚才定义loss_fn有关

            # print(y_pred.shape) # [batch_size, 10] # torch.Size([64, 10])
            # print(target.shape) # [batch_size] # torch.Size([64])
            # 计算准确率:
            # 按照第二维度来看, 也就是横向看
            accuracy = (y_pred.argmax(1) == target).float().mean()
            # print(y_pred)
            # print(y_pred.argmax(1))
            # break

            # 优化模型
            optimizer.zero_grad()  # 清空模型参数的梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 打印
            print("step:{}, loss:{}, acc:{} ".format(step, loss.item(), accuracy))

            if step % 100 == 0:
                if not os.path.isdir("./17_model"):  # 如果没有存在, 则创建该文件夹
                    os.makedirs("./17_model")
                # 保存模型, 并且在tensorboard中显示.
                # 每一轮的loss曲线都刻画出来
                writer.add_scalar("train_loss_epoch_{}".format(epoch), loss.item(), step)
                torch.save(model, "./17_model/model.pth")
                # .item(), 是当tensor对象中, 只有一个元素(数值)的时候,
                # .itme()是直接可以将其中的元素取出来, 不需要索引

            # break # 循环一轮, 看是否有错误.
        # 开始评估模式

        model.eval()  # 评估模式, 与上面同理
        total_accuracy = 0
        total_loss = 0
        test_length = test_dataloader.__len__()
        # 因为是评估模式, 所以我们不需要进行梯度的计算, 所以我们需要再no_grad中进行评估
        with torch.no_grad():
            for idx, (img, target) in enumerate(test_dataloader):
                # 评估模式也需要进行to操作.
                img = img.to(devise)
                target = target.to(devise)

                # 计算预测值
                y_pred = model(img)

                # 计算损失值
                loss = loss_fn(y_pred, target)

                # 计算准确率
                accuracy = (y_pred.argmax(1) == target).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy
        # 每epoch就写一次
        writer.add_scalar("eval_loss", total_loss, epoch)
        print("this is {} epoch, mean_loss: {}, mean_accuracy:{} ".format(i, total_loss / test_length,
                                                                          total_accuracy / test_length))

    writer.close()
    return None


if __name__ == '__main__':
    test01()
    # test_for_model()
    # accuracy_test()
