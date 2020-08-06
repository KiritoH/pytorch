import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

# 1/5 生成数据
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
# x0,y0 和 x1,y1分别对指两类数据
x0 = torch.normal(mean_value * n_data, 1) + bias
# 0类
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-mean_value * n_data, 1) + bias
# 1类
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

# 2/5 模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    # 设置前向传播(也就是计算结果过程)
    def forward(self, x):
        # 这就是设置模型过程,2和1是初始值,设置一个线性模型
        x = self.features(x)
        # 然后对线性模型放到sigmoid里面,就是所需要的模型
        x = self.sigmoid(x)
        return x

# 实例化逻辑回归模型
lr_net = LR()

# 3/5 选择损失函数(二分类交叉熵损失函数)
loss_fn = nn.BCELoss()

# 4/5 选择优化器(随机梯度下降法)
# momentum是啥?
lr = 0.01
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# 5/5 模型训练
for iteration in range(1000):
    # 前向传播
    y_pred = lr_net(train_x)

    #计算loss
    loss = loss_fn(y_pred.squeeze(), train_y)

    #反向传播
    loss.backward()

    #更新参数
    optimizer.step()

    # 绘图
    if iteration % 20 == 0:
        # 计算准确率
        mask = y_pred.ge(0.5).float().squeeze()    # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()     # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)    # 计算分类准确度

        # 绘制训练数据
        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break



