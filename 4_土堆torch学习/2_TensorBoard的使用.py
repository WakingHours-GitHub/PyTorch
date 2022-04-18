from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
"""
torch1.1版本后, 加入了TensorBoard
SummaryWriter类的使用
    writer = SummaryWriter(log_dir=)
        返回一个writer对象, 用于事件写入
    writer.add_scalar() # 增加标量数据
        注意, 这里tag注意区分, 否则将会在tensorboard显示的过程中图会乱
        
    writer.add_image()
    

如何打开我们生成的事件文件.
    tensorboard --logdir="path" --port=port
    
    eg: tensorboard --logdir="./logs" --port=6007
    
    
    
"""

# 初始化SummaryWriter对象
writer = SummaryWriter(log_dir="./logs")

for i in range(100):
    writer.add_scalar("y=x", i, i)
"""
    def add_scalar(
        self, 
        tag, # 图表的标题
        scalar_value, # 标量数据 -> 就是y轴
        global_step=None, # 多少步就是x轴
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
"""
writer.add_image("test",cv.imread("./hymenoptera_data/train/ants/0013035.jpg"), 1, dataformats="HWC")
"""
        tag (string): Data identifier # 数据标识符
        img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            # 这里输入的类型只能是Tensor, numpy.array, string 
            # numpy指的就是通过opencv进行读取的
            # 或者从PIL中读取的转换为ndarray类型, np.array(img)
            
        global_step (int): Global step value to record  # 步
        walltime (float): Optional override default walltime (time.time())
          seconds after epoch of event
      
        Shape:
        img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
        convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
        Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
        corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
        
        writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
"""


writer.close()

