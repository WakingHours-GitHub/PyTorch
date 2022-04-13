
"""
剩下的课程:
    模型保存与加载
    完成的训练套路
    实现使用GPU进行训练
    模型验证套路
    GitHub上面的优秀代码案例



模型的保存与加载
两种保存方式: 模型的保存和加载一定要对应


# 陷阱:
注意, 方式1加载时有一个陷阱, 就是保存完模型后, 再加载时
需要再加载的文件中, 也定义出这个class这个模型, 但是你无需实例化它
然后才可以加载模型
我们也可以再开头中直接引入: from xxx import *
所以在使用方式1的时候, 加载时我们应该让文件能够访问到我们定义的这个模型(class中)


"""
import torch
import torchvision

def test_load_model02():
    """
    对应保存方式2, 的加载模型
    返回的是ordereDict数据类型,
    :return:
    """
    vgg16 = torchvision.models.vgg16()
    model = torch.load("./16_model/vgg16_method2.pth")
    print(model)  # 打印的就是字典形式的参数

    # 那么第二种保存方式, 加载时, 就需要先实例化模型对象,
    # 然后使用.load_state_dict()这个函数, 将加载的参数字典, 放入到模型当中去.
    vgg16.load_state_dict(torch.load(""))
    #

def test_load_model01():
    """
    对应保存方式1, 来去加载模型
    :return:
    """
    #  之际加载一个模型
    vgg16 = torch.load("") # 读取网络模型, 返回的就是之前保存好的模型对象
    #


def test02():
    """
    模型的保存方式2
    只保存模型中的参数
    官方推荐, 因为这种保存方式只保存模型参数, 不保存网络结构
    :return:
    """
    vgg16 = torchvision.models.vgg16()
    torch.save(vgg16.state_dict(), "./16_model/vgg16_method2.pth")
    # 这种方式, 我们保存的是这个模型的状态字典, 也就是参数, 变成字典格式, 然后序列化到本地


def test01():
    """
    模型保存的方式1
    即保存模型的参数部分, 又保存模型的网络结构
    加载时返回的直接就是一个模型对象
    -> nn.Module
    :return:
    """
    vgg16 = torchvision.models.vgg16(
        pretrained=True,
    )
    torch.save(vgg16, "./16_model/vgg16_method1.pth")
    # 其中, 文件后缀名用什么都可以, 不过一般都是用pth
    # 这种方式即保存了模型的网络结构, 也保存了模型中变量的参数




if __name__ == '__main__':
    test01()










