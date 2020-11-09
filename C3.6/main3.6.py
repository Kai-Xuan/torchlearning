import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# print(sys.path)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)





