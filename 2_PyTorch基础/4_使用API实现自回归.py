import torch
from torch import nn # 导入网络组件
from torch import optim # 导入优化器对象


# 0. 准备数据:
x = torch.rand(size=(500, 1))
y_true = 3 * x + 0.8


# 1. 定义模型:
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__() # 继承父类的init
        self.Linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):  # 通过__call()__调用这个forward() 方法
        y_pred = self.Linear(x)

        return y_pred

# 2. 实例化模型, 优化器类实例化, loss实例化
my_linear = MyLinear()
sgd = optim.SGD(my_linear.parameters(), 0.001) # 将我们需要更新的参数列表传入进去
# my_linear.parameters() 返回的是一个生成器对象, 我们可以使用list转换回来, 然后通过索引来获取对象.
loss_fn = nn.MSELoss()


# 3. 循环, 进行梯度下降, 参数的更新:
for i in range(200):
    # 计算预测值
    y_pred = my_linear(x) # 对象当成函数调用, 直接使用
    loss = loss_fn(y_pred, y_true)

    # 重置梯度
    sgd.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    sgd.step()
    if i % 100 == 0:
        params = list(my_linear.parameters())
        print("loss:", loss.item(), params[0], params[1])




