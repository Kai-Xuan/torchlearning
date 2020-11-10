# matplotlib inline
import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    # d2l.plt.plot(x_vals.numpy(), y_vals.numpy())
    # if do this, # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()


# relu 函数示例
# relu 函数图
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
# relu 函数导数图
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


# sigmoid 函数示例
# sigmoid 函数图
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
# sigmoid 函数导数图
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


# tanh 函数示例
# tanh 函数图
y = x.sigmoid()
xyplot(x, y, 'tanh')
# tanh 函数导数图
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')






