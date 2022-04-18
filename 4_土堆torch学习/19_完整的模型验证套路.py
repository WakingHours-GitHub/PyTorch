"""

测试, demo:
核心: 其实就是利用已经训练好的模型, 提供输出, 得到输出, 进行应用.








"""

from PIL import Image
import torchvision.transforms as F
import torch

image_path = ""
image = Image.open()
print(img)  # Image对象

# 注意png是四个通道, 除了RGB三个通道之外, 还有一个透明度通道.
# 所以我们只能使用image = image.convent("RGB")保留其颜色通道
# 这样可与使用各种格式的图片

image = image.convert('RGB')
# 但是图片的格式与我们模型的输入格式不一样, 所以我们需要对其进行一个resize的操作
# torchvision.transform.Resize(size, interpolation)
# 参数: 输入的是PIL image, 或者是Tensor类型
# 返回一个PIL image或者是Tensor类型.

img2tensor = F.Compose([
    F.Resize((32, 32)),
    F.ToTensor()
])
# 这样就resize然后totensor了
image = img2tensor(image)
print(image.shape)
# 然后增加一层batch_size
image = torch.reshape(image, shape=(1, 3, 32, 32))
# 接着加载我们的模型
# 注意, 加载模型时, 该文件必须能够访问到模型类.

# 因为保存模型的都是cuda类型的, 所以在加载时, 我们需要映射到cpu上的类型
model = torch.load("", map_location=torch.device("cpu"))
model.eval()  # 开启评估模式
with torch.no_grad():  # 不追踪梯度, 可以节约我们的资源
    output = model(image)
print(output) # 输出预测结果
print(output.argmax(dim=1)) # 横向查找最大值的位置