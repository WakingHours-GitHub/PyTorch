"""
使用GPU训练我们的模型:
    torch.cuda.is_available()
    判断该机器是否支持GPU.

我们可以使用谷歌官方提供的免费GPU来进行训练
官网: google colab
可以设置: 文件: 图形硬件加速器
如果在笔记中使用终端, 可以在前面加上!

一共有两种方式:
    第一种:
        网络模型
            对象.cuda() 返回新的cuda对象
        数据(输入, 标注)
            从dataloader中取出来的
        损失函数
        (注意, 优化器不需要.cuda())

        .cuda() 返回的就是GPU上的tensor对象

    第二种: (更加常用)
        device = torch.device("cuda")
        如果你的计算机上有多个显卡, 那么你可以指定你的显卡
        torch.device("cuda:0") # 指定第一个显卡
        .to(device)

        网络模型
        损失函数
        以及输入,和label
        都需要.to(device).

    一些细节, 模型, loss_fn, 都不需要接收.cuda()和.to()的结果
    只有变量和标签, 需要接收结果

"""