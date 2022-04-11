import torch
from torch import nn
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device) # cuda:0

# 准备数据:
x = torch.rand(size=(500, 1)).to(device)
y_true = 3 * x + 0.8 # 由x构建出来的y, 肯定也是cuda类型的

# 创建模型
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__() # 继承
        self.Linear = nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.Linear(x)

        return y_pred

# 实例化模型
my_linear = MyLinear().to(device)
optim_sgd = optim.SGD(my_linear.parameters(), 0.01)
loss_fun = nn.MSELoss().to(device)


# 3. 循环, 进行梯度下降, 并且更新参数
for i in range(5000):
    # 计算y_pred
    y_pred = my_linear(x)
    loss = loss_fun(y_pred, y_true)

    # 重置梯度:
    optim_sgd.zero_grad()

    # 反向传播
    loss.backward()


    # 更新参数
    optim_sgd.step()
    if i % 100 == 0:
        params = list(my_linear.parameters())
        print("loss:", loss.item(), params[0], params[1])


