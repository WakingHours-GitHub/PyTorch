from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter # 导入日志

"""
让我们来看看这个设置的参数:
def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
             shuffle: bool = False, sampler: Optional[Sampler] = None,
             batch_sampler: Optional[Sampler[Sequence]] = None,
             num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
             pin_memory: bool = False, drop_last: bool = False,
             timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
             multiprocessing_context=None, generator=None,
             *, prefetch_factor: int = 2,
             persistent_workers: bool = False):
             
    - dataset(Dataset): 就是我们实例化的dataset对象, 直接放入到这里
    - batch_size(int): batch分组的长度. 一组数据.
    - shuffle(bool): shuffle, 打乱顺序读取
    - num_workers(int): 使用多少个进程进行加载, >0时在windows platform下, 有时候会BrokenPipeError,
                        此时需要将num_workers设为0
    - drop_last(bool): 最后的batch组, 可能不能达到batch_size, 所以我们可以去掉最后一组.
    
    


"""
def test03():
    CIFAR10_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    CIFAR10_dataloader = DataLoader(
        dataset=CIFAR10_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2, # 使用两个线程进行批处理
        drop_last=True
    )
    # 通过循环两次
    with SummaryWriter("6_logs") as writer:
        for epoch in range(2): # 这里就是总共两次, 遍历dataloader
            # 这里一次循环, 就是遍历一次dataloader.
            for idx, (img, label) in enumerate(CIFAR10_dataloader):
                writer.add_images(f"{epoch} test_img", img, idx)
    # 在tensorboard中, 发现, 当shuffle=False时, 两次batch结果是一样的,
    # 当shuffle=True时, 两次batch的顺序就不一样. 可能对训练效果产生更好的影响.

    pass

def test02():
    # 加载数据集
    CIFAR10_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    CIFAR10_dataloader = DataLoader(
        dataset=CIFAR10_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2, # 使用两个线程进行批处理
        drop_last=False # 这里体现出
    )

    with SummaryWriter("6_logs") as writer:
        # 注意(img, label)需要加上括号, 因为返回的是一个idx和元素, 类型, 所以如果不加括号就是三个变量进行接受了
        for idx, (img, label) in enumerate(CIFAR10_dataloader):
            # 注意在对batch后的图片显示到tensorboard时, 需要使用add_images
            writer.add_images("test_img", img, idx)

    CIFAR10_dataloader_False = DataLoader(
        dataset=CIFAR10_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    with SummaryWriter("6_logs") as writer:
        for idx, (img, label) in enumerate(CIFAR10_dataloader_False):
            writer.add_images("test_img_droplast_True", img, idx)

    # 在tensorboard中可以看到, 没有加drop_last=False, 最后一组的图片是不完整的只有16张图片
    # 而drop_last=True, 的可以发现, 只有九组batch数据, 最后一组被删掉了








def test01():
    # 准备的测试数据集
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10_data",
        train=False, # 取训练集, 这样数据较少
        transform=torchvision.transforms.ToTensor()
    )

    CIFAR10_test_loader = DataLoader(
        dataset=CIFAR10_test_dataset, # 设置dataset
        batch_size=4, # 4个为一组
        shuffle=True, #
        num_workers=0,
        drop_last=False
    )

    # 测试数据的第一张图片和对应标签
    img_tensor, label = CIFAR10_test_dataset[0]
    print("img_tensor.shape: ", img_tensor.shape) # img_tensor.shape:  torch.Size([3, 32, 32])
    print("label:", label) # label: 3

    # 查看数据集加载器:
    # 这一轮for就是遍历一次dataloader.
    # 如果使用两轮循环, 那么此时shuffle就起到作用了.

    for img, label in CIFAR10_test_loader:
        print("img.shape: ", img.shape) # img.shape:  torch.Size([4, 3, 32, 32])
        print("label: ", label) # label:  tensor([7, 0, 6, 8])
        break











if __name__ == '__main__':
    # test01()
    # test02()
    test03()