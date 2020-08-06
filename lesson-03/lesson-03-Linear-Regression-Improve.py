import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)

# 学习率
lr = 0.01
# 存储当前最好的损失
best_loss = float("inf")

# 创建训练数据
x = torch.rand(200, 1) * 10
y = 3*x + (5 + torch.randn(200, 1))

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(10000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算MSE loss(损失函数)
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数(要用学习率乘以梯度,以免梯度爆炸)
    w.data.sub_(lr * w.grad)
    b.data.sub_(lr * b.grad)

    current_loss = loss.item()
    # best_loss仅用来记录而已
    if current_loss < best_loss:
        best_loss = current_loss
        best_w = w
        best_b = b

    # 绘图
    if iteration % 100 == 0:
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
        # 如果损失函数值小于0.55时停止
        if loss.data.numpy() < 0.55:
            break

    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

# 输出一下best_loss
print(best_loss, best_w, best_b)