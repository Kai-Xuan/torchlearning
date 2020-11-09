import torch
import numpy as np
import random

num_inputs = 2
num_examples = 1000
true_w = [3,2.2]
true_b = 2.5
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

print(features,labels)

def data_iter(batch_size, features, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)]) # 最后一次可能不足一个batch
        yield features.index_select(0,j), labels.index_select(0,j)

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
    # break

def linereg(X, w, b):
    return torch.mm(X,w) + b

def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size  # 注意这里更改param时用的param.data, but why？

lr = 0.03
num_epochs = 30
net = linereg
loss = square_loss
w = torch.tensor(np.random.normal(0, 0.01, [num_inputs,1]), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features,w,b),labels)
    print('epoch: %d, loss: %f' % (epoch+1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)




















