

import torch
import torch.nn as nn


net = nn.Linear(3,1)
print(list(net.parameters())[0].device)

net1 = nn.Linear(3,1).cuda()
print(list(net1.parameters())[0].device)

# 同样的，我么需要保证模型输入的 Tensor 和 模型 都在同一设备上，否则会报错。
x = torch.rand(2,3).cuda()
net1(x)

