#from https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.1_model-construction
#copyright @ https://tangshusen.me/Dive-into-DL-PyTorch/#/
import torch
import torch.nn as nn


class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwards):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP,self).__init__(**kwards)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

# x = torch.rand(2, 784, requires_grad=True)
x = torch.rand(2, 784)
net = MLP()
print(net)
y = net(x)
y.sum().backward()
print('testing')


