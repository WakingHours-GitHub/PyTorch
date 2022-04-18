from torchvision import transforms
from torchvision.transforms import  *
import cv2 as cv



"""
transforms的结构与用法
    transforms.py -> 工具箱.
    
在python中的用法:
通过transforms.ToTensor去解决两个问题
    1. transforms应该如何使用
        首先实例化对象, 然后使用该对象当成函数调用__call__方法, 
        对input进行处理, 返回output
    2. Tensor数据类型, 是什么?
        带有一系列属性的存储容器, 是一个高维数据结构
"""
# 使用cv2进行读取文件, ndarray数据类型
img = cv.imread("./hymenoptera_data/train/ants/0013035.jpg")

# 创建对象,
toTensor = transforms.ToTensor() # 返回一个实例
# 然后我们直接调用他的__call()__方法
img_tensor = toTensor(img)
print(img_tensor)




