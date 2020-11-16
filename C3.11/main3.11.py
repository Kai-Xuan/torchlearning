# 模型选择 欠拟合，过拟合
# 欠拟合： 模型无法得到较低的训练误差
# 过拟合： 模型得到训练误差远小于它在测试数据集上的误差
# 给定训练数据集，如果模型的复杂度过低，很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。
# https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.11_underfit-overfit?id=_3114-%e5%a4%9a%e9%a1%b9%e5%bc%8f%e5%87%bd%e6%95%b0%e6%8b%9f%e5%90%88%e5%ae%9e%e9%aa%8c

#matplotlib inline
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

n_train, n_test, true_w, true_b = 100, 100, [1,2, -3.4, 5.6], 5
features = torch.randn((n_train+n_test, 1))
## 起初在这里 我使用torch.rand函数，并无法得到预期的演示效果

poly_features = torch.cat((features, torch.pow(features,2), torch.pow(features,3)), dim=1)
labels = true_w[0] * poly_features[:,0] + true_w[1] * poly_features[:,1] + true_w[2] * poly_features[:,2] + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
labels += torch.normal(0, 0.01, size=labels.size(), dtype=torch.float)

num_epochs, loss = 100, torch.nn.MSELoss()


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)

    d2l.plt.show()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1],1)

    batch_size = min(10, train_labels.shape[0])

    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X),y.view(-1,1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


        train_labels = train_labels.view(-1,1)
        test_labels = test_labels.view(-1,1)

        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())

    print('final epoch: train_loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])


fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])


fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])