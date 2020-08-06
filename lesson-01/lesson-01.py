import numpy as np
import torch
'''张量的创建'''

'''第一类:直接创建'''
# example 1
# 通过torch.tensor创建张量
# 测试时设一个flag,测试哪个开哪个
flag = False
if flag:
    # arr相当于data,tensor只是对其进行了封装
    arr = np.ones((3, 3))
    print("ndarray的数据类型", arr.dtype)
    # 如果要gpu,后面参数可以用'cuda',由于没有gpu,所以写'cpu'
    t = torch.tensor(arr, device='cpu')
    print(t)


# example 2
# 通过torch.from_numpy创建张量
flag = True
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("arr's id is {}, and t's id is {}".format(id(arr), id(t.data)))
    print(t)
    # 通过这个方法得出的tensor,其data和ndarray是同一个数组(引用)
    arr[1, 1] = 2000000
    print(arr)
    print(t)

'''第二类: 依据数值创建'''
# example 3
# 通过torch.zeros创建张量
flag = False
if flag:
    # 随便创一个张量,用于接收数据
    out_t = torch.tensor([1])
    # out是指输出的张量(用于接收输出的张量)
    # zeros是一个全零张量
    t = torch.zeros((3, 3), out=out_t)
    print(t, '\n', out_t)
    # t和out_t是一样的..
    print(id(t), id(out_t), id(t) == id(out_t))

# example 4
# 通过torch.full创建全10张量
flag = False
if flag:
    # 由于现在的pytorch版本更新,如果不指定张量中数据类型
    # 会默认为bool类型,所以需要自己指定,否则会报错
    t = torch.full((3, 3), 10, dtype=float)

    print(t)

# example 5
# 通过torch.arange创建张量
flag = False
if flag:
    # 第三个参数是指步长
    t = torch.arange(2, 10, 2)
    print(t)

# example 6
# 通过torch.linspace创建张量
flag = False
if flag:
    # 第三个参数是指分成几份
    t = torch.linspace(2, 10, 6)
    print(t)


# example 7
# 通过torch.normal创建正态分布张量
flag = False
if flag:
    # mean: 张量; std: 张量
    # 这个相当于mean和std一一对应得到多个分布一一采样得到一组值
    mean = torch.arange(1, 5, dtype=torch.float)
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)

    # mean: 标量; std:标量
    # 这个相当于对于一个分布,取一组值(需要指定大小)
    t_normal = torch.normal(0., 1., size=(4,))
    print(t_normal)

    # mean: 张量; std:标量
    # 这个相当于,对于多个分布,期望分别取mean中各个值,方差均为std,每个采样一次,得到一组值
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)
    print(t_normal)

    # mean: 标量; std:张量
    # 这个相当于,对于多个分布,期望均为mean,方差取std中各个值,每个采样一次,得到一组值
    mean = 1
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print(t_normal)




