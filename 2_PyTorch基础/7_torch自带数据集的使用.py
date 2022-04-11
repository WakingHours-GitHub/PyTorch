import torchvision
import torch


# 导入数据集
# 先下载数据集
# mnist = torchvision.datasets.MNIST(root="./mnist_data", download=True)
mnist = torchvision.datasets.MNIST(
    root="./mnist_data",
    train=True, #
    transform=None # 用于对数据集进行的操作
)
print(type(mnist))  # <class 'torchvision.datasets.mnist.MNIST'>
# 实际上是mnist也是继承DataSets的
print(mnist)  #
# Dataset MNIST
#     Number of datapoints: 60000
#     Root location: ./mnist_data
#     Split: Train

# 接下来看一下该数据集对象中都是什么
print(mnist[0])  # (<PIL.Image.Image image mode=L size=28x28 at 0x16AD4F1AFD0>, 5)
# 这实际上, 就对应了,(内容, 标签)
mnist[0][0].show()  # Image中的, 显示图片
