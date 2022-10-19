import torch

# 打印touch版本
print("PyTorch version: ", torch.__version__)

# 查看是否支持GPU
print("GPU:", torch.cuda.is_available())


