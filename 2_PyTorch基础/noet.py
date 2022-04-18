"""
笔记:
    在PyTorch中, 所有的数据类型都可以对应起来
    从常规类型 -> 到Tensor类型
    但是PyTorch中的string类型没有对应的Tensor类型
    那么在PyTorch中如何表示string呢?
    一种是one_hot编码表示, 常用于文本较少,(类别较少的情况)
    还有一种是Embedding编码, 也就是一种用数字来表示字符串

    Data Type:
        data type       dtype           cpu tensor          gpu tensor
        32bit float     torch.float32,  torch.FloatTensor   torch.cuda.FloatTensor
                        torch.float
        64bit float     torch.float64,  torch.DoubleTensor  torch.cuda.DoubleTensor
                        torch.double
        8bit integer    torch.uint8     torch.ByteTensor    torch.cuda.ByteTensor
        unsigned

Dimension 0:　就是常量:
    torch.tensor(数)
    在计算loss中, 我们最终得到的值就是一个常量
Dim 1: 向量: []
    torch.tensor([数据]) # 直接是现有的数据作为张量
    torch.Tensor(shape) # 就是随机初始化的一个shape的Tensor
    上面这两点我们要区分出来.当然,Tensor也可以接受数据,只不过需要用[]包裹起来, 一般我们不使用

    或者从numpy:
    torch.from_numpy(ndarray对象)
    如果是浮点型, 导入到torch中就是float64类型

    偏置: bias就是一个一阶的Tensor,
    Linear input, 线性回归的输入, 也是一个一阶Tensor
Dim 2: 二阶张量, 矩阵: [[], []...]
    这种数据类型用过的最多的就是:
        Linear input barch, 也就是输入组, 一组的输入.
Dim 3: 三阶张量
    RNN input batch -> 适用于文字处理
Dim 4: 四阶张量
    适用于CNN: 存储图片batch
    [b, c, h, w]
额外的知识:
    .numel() 是指tensor所占用内存的数量


创建Tensor:
    创建一些普通的Tensor:
        torch.tensor([数据]) # 直接是现有的数据作为张量
        torch.Tensor(shape) # 就是随机初始化的一个shape的Tensor
    未初始化: 的变量
        torch.empty() # 生成一个空的未初始化的变量
        torch.FloatTensor()

    使用未初始化的的Tensor会出现什么问题.
    数据是非常混乱的. 有非常大,或者非常小的数据.
    注意, 在使用未初始化的Tensor时, 我们一定要覆盖掉

设置默认类型: set default type:
    torch中默认的类型是: FloatTensor
    不过我们可以设置: torch的默认类型
    torch.set_default_tensor_type(torch.DoubleTensor)

随机初始化:
    torch.rand(shape) -> 会产生0~1之间的数值, 不包括1 (符合均匀分布)
    如果要采样到[min, max)的rand:
    那么就是: (max-min)
    *_like(a) 就是产生和a同shape的张量

    torch.randint(min, max, shape) -> 会产生一个[min, max)的shape形状的随机整数

    torch.randn(shape) -> 随机标准正态分布初始化

    当然也有正态分布初始化:
    torch.normal(maen=就是每个元素的均值, std=就是每个元素的标准差)

    例子:
        torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
        就是生成一维, 10个元素的, 每个元素符合mean=0, 和std逐渐建减小的正态分布

    torch.full(shape, 值) # 将shape的Tensor全部赋值成一个元素
    torch.arange(a, b, steps) # 类似与range, [a, b)
    torch.range(a, b, steps) # 就是[a, b] 不过不推荐使用

    linspace(a, b, steps) # 就是生成[a, b]均匀分割steps个点

    ones/zeros/eye快速生成特定的张量.

    *_like(Tensor) 就是生成Tensor同形状的一个张量

     randperm(10) # 就是生成随机索引(序列), 达到协同shuffle的一个功能

索引,切片:
















"""


