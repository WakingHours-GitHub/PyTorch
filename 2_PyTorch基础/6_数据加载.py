import math

import torch
from torch.utils.data import Dataset, DataLoader  # 导入数据加载包

# 数据集来源: http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
data_path = "./SMSSpamCollection"  # 保存的数据来源路径


# 完成数据集类:
class SMSS_dataset(Dataset):  # 继承Dataset父类
    def __init__(self):  # 构造函数, 直接执行
        # 初始化每个对象都有一个lines对象属性
        # 这里就是将数据文件读取进来, 这里需要根据数据集的特点来自定义
        self.lines = open(data_path, 'r', encoding="utf-8").readlines()  # 获取所有行
        # [line, line, ...]

    # 这就是容器类 -> 魔术方法:
    # 凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样
    # p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
    def __getitem__(self, key):  # 根据索引获取数据
        cur_line = self.lines[key].strip()  # 去除空格, 和换行
        # 切分成label和feature
        label = cur_line[0:4].strip()  # 标签
        content = cur_line[4:].strip()  # 内容
        return label, content  # 返回

    def __len__(self):  # 返回数据的总数量
        return len(self.lines)


# 练习:
# class s


def test01():
    """
    使用继承dataset() 子类. 生成数据集类

    :return:
    """
    my_dataset = SMSS_dataset()  # 实例化对象
    # 这里就已经触发了init方法, 将数据读取进来
    print(my_dataset[2])  # 直接调用回调函数, 返回取回的值
    print(len(my_dataset))  #

    # 我们也可以进行遍历：
    for i in range(len(my_dataset)):
        print(my_dataset[i])


def test02():
    # 实例化数据集对象
    data_set = SMSS_dataset()

    dataloader = DataLoader(  # batch批处理. 返回批处理队列.
        dataset=data_set,  # 数据集对象
        batch_size=2,  # batch, 分组大小
        shuffle=True,  # 是否打乱
        num_workers=2,  # 线程数
        # drop_last=True # 是否删除最后一个batch
        # 如果最后, dataloader中最后一个batch_可能不是完整的, 那么此时进行训练, 就会出现错误.
        # 所以我们可以加一个参数: drop_last: 就是把最后一个batch删掉, 以确保,batch的完整性
    )
    # 返回的数据：
    # [(,...), (,...)] # 根据你的batch_size有关

    for dl in dataloader:  #
        print(dl)
        # [('ham', 'spam'), ('Your opinion about me? 1. Over 2. Jada 3. Kusruthi 4. Lovable 5. Silent 6. Spl character 7. Not matured 8. Stylish 9. Simple Pls reply..', 'Your free ringtone is waiting to be collected. Simply text the password "MIX" to 85069 to verify. Get Usher and Britney. FML, PO Box 5249, MK17 92H. 450Ppw 16')]
        break

    # 我们经常会使用 enumerate, 就是, 将可迭代对象与index共同返回.
    for index, (label, content) in enumerate(dataloader):
        print(index, label, content)  #
        break

    print(len(data_set)) # 5574
    print(len(dataloader)) # 2787
    # 如果除不开, 就像上取整
    print(math.ceil(len(data_set)/2))


def test03():
    pass
    #

# 测试：
if __name__ == '__main__':
    test02()
