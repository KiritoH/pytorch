import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

# 计算
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# backward是使梯度反向传播
# 这个方法算出了y对所有在计算图中的参数的导数(即反向传播)
# 然后各个张量可以通过grad属性得到y对各自的导数值
y.backward()
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度,非叶子结点的梯度都会被释放掉
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
# grad_fn用于记录创建该张量时所需要的计算方法
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)