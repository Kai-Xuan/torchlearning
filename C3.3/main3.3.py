import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data # 分batch_size 加载训练数据用的
import torch.optim as optim # 优化器



num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)

# labels = torch.matmul(features, torch.tensor(true_w)) + true_b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels = labels + torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

batch_size = 10

dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
nn.init.normal_(net.linear.weight, mean=0, std=0.01)
nn.init.constant_(net.linear.bias, val=0)
# print(net)
# >>
# LinearNet(
#   (linear): Linear(in_features=2, out_features=1, bias=True)
# )

# print(net.linear.weight)
# >>
# Parameter containing:
# tensor([[0.0010, 0.0052]], requires_grad=True)

# print(net.linear.bias)
# >>
# Parameter containing:
# tensor([0.], requires_grad=True)

# print(len(net.parameters()))

loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


print(true_w, net.linear.weight)
print(true_b, net.linear.bias)








