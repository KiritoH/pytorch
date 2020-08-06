import torch
torch.manual_seed(10)

# retain_graph
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    # 实际上这个backward就是torch.autograd.backward
    # y.backward()
    print(w.grad)
    # 由于pytorch采用动态图机制,在一次求导后(backward)就会清除之前的计算图
    # 如果想要多次使用该计算图,可以在backward中加上属性"retain_graph = True"
    y.backward(retain_graph=True)

# grad_tensors,用于设置多个梯度之前的权重
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y0 = torch.mul(a, b)
    y1 = torch.add(a, b)
    # 拼接
    loss = torch.cat([y0, y1], dim=0)
    # 设置梯度的权重
    grad_tensor = torch.tensor([1., 2.])
    # 这个gradient对应了torch.autograd.backward()中的grad_tensor
    loss.backward(gradient=grad_tensor)
    # 如此结果为9
    print(w.grad)

# autograd.grad
# 这里实现一个二阶求导
# flag = True
flag = False
if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)
    # 一定要设置为True,才能对其进行下一次求导
    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)
    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

# 梯度不会自动清零
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        # 这里会不断往上增加,所以需要手动对梯度清零
        print(w.grad)
        # 有一个下划线代表了原地(in place)操作
        w.grad.zero_()

# 依赖于叶子结点的结点,requires_grad默认为True
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# 叶子结点不可执行in place操作(比如add_, zero_ 等等)

# 什么是in place操作?
flag = True
# flag = False
if flag:
    a = torch.ones((1, ))
    print(id(a), a)
    # a = a + torch.ones((1, ))
    # print(id(a), a)
    a += torch.ones((1, ))
    print(id(a), a)


flag = True
# flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 会报错
    w.add_(1)
    """
    autograd小贴士：
        梯度不自动清零 
        依赖于叶子结点的结点，requires_grad默认为True     
        叶子结点不可执行in-place 
    """
    y.backward()