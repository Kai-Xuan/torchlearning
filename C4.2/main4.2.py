
# https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.2_parameters

import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()


# 访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
# <class 'generator'>
# 0.weight torch.Size([3, 4])
# 0.bias torch.Size([3])
# 2.weight torch.Size([1, 3])
# 2.bias torch.Size([1])

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))
# weight torch.Size([3, 4]) <class 'torch.nn.parameter.Parameter'>
# bias torch.Size([3]) <class 'torch.nn.parameter.Parameter'>


