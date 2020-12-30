
# from https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.5_read-write
# 读取和存储

# 下面的例子创建了Tensor变量x，并将其存在文件名同为x.pt的文件里

import torch
from torch import nn


# 保存数据
x = torch.ones(3)
torch.save(x, 'x.pt')
# 读取数据
x2 = torch.load('x.pt')
print(x2)


# 存储和读取列表
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)


# 存储并读取一个从字符串映射到Tensor的字典。
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


# 4.5.2 为保存和加载模型
# 相信参考 https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.5_read-write


