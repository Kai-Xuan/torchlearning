

# from https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.6_use-gpu

import torch
from torch import nn

print(torch.cuda.is_available()) # 输出 True


# 查看GPU数量：
print(torch.cuda.device_count())
# 查看当前GPU索引号，索引号从0开始
print(torch.cuda.current_device()) # 输出 0
# 根据索引号查看GPU名字:
print(torch.cuda.get_device_name(0)) # 输出 'Quadro P5000'

x = torch.tensor([1, 2, 3])
print(x)
# 使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，我们用.cuda(i)来表示第 ii 块GPU及相应的显存（ii从0开始）且cuda(0)和cuda()等价。
x = x.cuda(0)
print(x)

# 可以通过Tensor的device属性来查看该Tensor所在的设备。
print(x.device)

# 可以直接在创建的时候就指定设备。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)
