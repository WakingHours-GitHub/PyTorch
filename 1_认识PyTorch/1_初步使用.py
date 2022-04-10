import torch

# 打印touch版本
print("PyTorch version: ", torch.__version__)

# 查看是否支持GPU
print("GOU:", torch.cuda.is_available())
