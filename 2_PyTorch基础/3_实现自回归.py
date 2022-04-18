import torch
import matplotlib.pyplot as plt

learn_rate = 0.01

## 思路:
# 1. 准备数据
x = torch.rand((500, 1))  # 随机创建变量
y_true = x * 3 + 0.8
# y_true = torch.add(torch.mul(x, 0.3), 0.8)


# 2. 通过模型计算y_pred
# 因为w, b是模型参数, 所以我们需要设置: requires_grad=True
w = torch.randn([1, 1], requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
# RuntimeError: Only Tensors of floating point and complex dtype can require gradients
# 运行错误, 只有这个Tensor类型是float或者更复杂的类型才可以使用requires_grad=True, 所以我们要不指定dtype, 要么.0
# 利用w,b计算y_pred
# y_pred =torch.add(torch.matmul(w, x), b)


# 3. 计算loss
# 回归问题的损失函数, 实际上是SSE
# loss = (y_true - y_pred).pow(2).mean() # sum(pow(y_true-y_pred  , 2))


# 4. 通过循环, 反向传播, 更新参数

for i in range(2000):
    # 3. 计算loss  ->  要不断计算loss, 以此来优化参数
    # 回归问题的损失函数, 实际上是SSE
    y_pred = torch.matmul(x, w) + b
    loss = (y_true - y_pred).pow(2).mean()  # sum(pow(y_true-y_pred  , 2))
    # 必须将梯度重置为0, 这是与TF的区别
    if w.grad is not None:  # 如果不是None
        w.grad.zero_()  # 清零
    if b.grad is not None:
        b.grad.zero_()

    # 反向传播:
    loss.backward()
    # 更新: 
    w.data = w.data - learn_rate * w.grad
    b.data = b.data - learn_rate * b.grad
    print("w:", w.item(), "\tb:", b.item(), "\tloss:", loss.item())

# 利用plt进行画图:
plt.figure(figsize=(20, 8))
# 画出散点图
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
y_pred = torch.add(torch.matmul(x, w), b)
plt.plot(x.numpy().reshape(-1), y_pred.detach().numpy().reshape(-1), color="red")

plt.show()


