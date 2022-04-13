import torchvision
from torch.utils.tensorboard import SummaryWriter # 导入日志写入器
from torchvision import datasets
from torchvision.transforms import transforms
"""
torchvision中自带的数据集的使用 
from torchvision import datasets
API: dataset.*()
    root: str, # 数据路径
    train: bool = True, # 是否训练
    transform: Optional[Callable] = None, # 对x进行transform操作的函数.
    target_transform: Optional[Callable] = None, # 对标签需要进行的transform
    download: bool = False, # 是否下载, 如果为True, 则默认下载到root下的路径

返回的是datasets:

自带datasets和transforms的联合使用
    从torchvision中加载的数据集往往不是一个tensor数据类型, 所以我们需要使用transforms
    对输出的数据进行转换, 而在torchvision中自带的数据集, 有一个transform参数, 就是我们放
    transforms函数的组合
"""
CIFAR10_root = "./CIFAR10_data"
# 首先先下载数据集:
torchvision.datasets.CIFAR10(
    root="./CIFAR10_data",
    train=True,
    download=True # 下载, 并解压
)


def test02() -> None: # 限制, 函数返回的数据类型为None
    CIFAR10_datasets = datasets.CIFAR10(
        root=CIFAR10_root, # 路径名称
        train=True,
        transform=transforms.Compose([  # 对数据进行处理
            transforms.ToTensor(), # 这里将PIL.Image -> torch.Tensor类型
        ]),
        download=True
    )
    print(CIFAR10_datasets.classes)
    img, target = CIFAR10_datasets[0]

    print(type(img)) # <class 'torch.Tensor'>]
    # 此时在datasets中, 图片就已经转换成Tensor对象了.
    print(CIFAR10_datasets.classes[target])

    # 上传到tensorboard中, 每个图片
    with SummaryWriter("logs") as writer:
        for i in range(10):
            img, _ = CIFAR10_datasets[i] # 这里省略target
            writer.add_image("test_img", img, i)


    return None




def test01() -> None:
    """
    讲解torchvision中的
    :return:
    """
    CIFAR10_train_datasets = torchvision.datasets.CIFAR10(
        root=CIFAR10_root,
        train=True,
        download=False
    ) # -.> 返回的是一个datasets对象

    print(CIFAR10_train_datasets[0]) # (<PIL.Image.Image image mode=RGB size=32x32 at 0x1EAA7053BB0>, 6)
    img, target = CIFAR10_train_datasets[0]


    # 返回所有类别
    print(CIFAR10_train_datasets.classes) # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    img, target = CIFAR10_train_datasets[0]
    img.show()
    print("target: ", CIFAR10_train_datasets.classes[target])








if __name__ == '__main__':
    # test01()
    test02()
