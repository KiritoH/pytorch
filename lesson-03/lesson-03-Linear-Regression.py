import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)

# 学习率
lr = 0.05

# 创建训练数据
x = torch.rand(20, 1) * 10
y = 2.2*x + (5 + torch.randn(20, 1))

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算MSE loss(损失函数)
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 方向传播
    loss.backward()

    # 更新参数(要用学习率乘以梯度,以免梯度爆炸)
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:
        # 画点
        plt.scatter(x.data.numpy(), y.data.numpy())
        # 画线
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        # 写上损失函数mean值
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        # x,y轴的间距?
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        # 标题,显示x和y的值
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        # 如果损失函数值小于1时停止
        if loss.data.numpy() < 1:
            break

