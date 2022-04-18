from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import cv2 as cv

# from PLT import image 我没下这个包
"""

常见的Transforms

1. transforms.Compose([])
    将多个transforms函数组合, 依次执行
2. transforms.ToTensor()
    将,PLT.image, ndarray类型的数据转换为Tensor类型
3. transforms.ToPILImage()
    将我们的Tensor对象转换为以一个PLT.image对象
4. transforms.Normalize()
    归一化tensor image, 使用mean和std对输入的tensor image进行一个归一化处理
    注意, mean, std应该和通道数同维度
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
5. transforms.Resize()
    resize输入的PLT image 到给定的size大小, 注意, 只能使用PLT image类型, 不能使用其他类型
    如果给定是一个序列:(h, w)那么input就会缩放到这个大小
    如果只给定了一个数, 那么就按照最短边缩放到这个数值

总结使用方法:
    关注输入和输出的类型
    注意API
    注意方法的参数, 以及使用方法
    是先实例化对象, 然后再直接调__call__方法
    学会再tensorboard中的显示


"""

def test03():
    writer = SummaryWriter("./logs")
    img = Image.open("./image.jpg")
    img_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])(img)
    print(img_tensor.dtype)


def test02():
    """
    Resize()使用
    :return:
    """
    write = SummaryWriter(log_dir="./logs")
    # img = cv.cvtColor(cv.imread("./image.jpg"), cv.COLOR_BGR2RGB)
    # img_resize = transforms.Resize((512, 512))(img)
    # print(img_resize.shape)
    img = Image.open("./image.jpg")
    print('img.size:', img.size) # img.size: (222, 169)
    img_resize = transforms.Resize((512, 512))(img)
    print('img_resize', img_resize.size) # img_resize (512, 512)
    print(img_resize)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x248FC1E1E80>
    # 可见转换完的数据仍然是一个PIL数据类型, 所以我们需要使用ToTensor进行转换
    img_tensor = transforms.ToTensor()(img_resize)
    write.add_image("resize_tensor", img_tensor)

    write.close()

    # 可见我们是先Resize()然后又经过了ToTensor, 此时我们就可以使用Compose进行组合
    # 见test03().





def test01():
    """
    ToTensor()和Normalize(0的使用

    :return:
    """
    write = SummaryWriter(log_dir="./logs")
    img = cv.imread("image.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print(type(img))  # <class 'numpy.ndarray'>
    img_tensor = transforms.ToTensor()(img)
    print(img_tensor)
    print(type(img_tensor))  # <class 'torch.Tensor'>
    write.add_image(tag="ToTensor_image", img_tensor=img_tensor)

    # 进行归一化处理:
    img_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])(img_tensor)
    # ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    print(img_norm)
    write.add_image("normalize", img_norm)

    write.close()


if __name__ == '__main__':
    # test01()
    test02()