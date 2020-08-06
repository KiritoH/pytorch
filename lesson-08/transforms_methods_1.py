import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt

path_lenet = os.path.join(BASE_DIR, "..", "model", "lenet.py")
path_tools = os.path.join(BASE_DIR, "..", "tools", "common_tools.py")
assert os.path.exists(path_lenet), "{}不存在，请将lenet.py文件放到 {}".format(path_lenet, os.path.dirname(path_lenet))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

# ????
from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


# ============================ step 1/5 数据 ============================
# 设置数据的路径
split_dir = os.path.join(BASE_DIR, "..", "..", "data", "rmb_split")
if not os.path.exists(split_dir):
    raise Exception(r"数据 {} 不存在, 回到lesson-06\1_split_dataset.py生成数据".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(196),

    # 2 RandomCrop
    # 随机填
    # transforms.RandomCrop(224, padding=16),
    # 自定左右上下填
    # transforms.RandomCrop(224, padding=(16, 64)),
    # 填充区域设置颜色
    # transforms.RandomCrop(224, padding=(16, 64), fill=(255, 0, 0)),
    # 如果size大于原图像尺寸时,必须价格pad_if_needed参数
    # transforms.RandomCrop(512, pad_if_needed=True, padding=(16, 64)),
    # 由边缘决定填充颜色
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # 镜像填充， 最后一个像素不镜像
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # 镜像填充， 最后一个像素镜像
    # transforms.RandomCrop(1024, padding=512, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # 随机大小,长宽比裁剪图像
    # transforms.RandomResizedCrop(size=224, scale=(0.08, 1)),

    # 4 FiveCrop
    # 在图像的上下左右以及中心裁剪出尺 寸为 size 的 5 张图片，
    # 对返回图像需要转换格式
    # transforms.FiveCrop(112),
    # lambda是匿名函数,冒号之前为函数输入,冒号之后为函数返回值
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # TenCrop 对这 5 张图片 进行水平或者垂直镜像获得 10 张图片
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # 水平翻转,p为1是必定翻转
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # 垂直翻转,p为0.5是一半概率翻转
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # 旋转一个角度在(-90, 90)中间
    # transforms.RandomRotation(90),
    # expand为是否扩大图片,保持原图信息
    # transforms.RandomRotation((90), expand=True),
    # center为旋转中心
    # transforms.RandomRotation(30, center=(0, 0)),
    # 开启expand只对绕中心旋转有效,其他中心无法做到全部显示
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):
        inputs, labels = data  # B C H W

        img_tensor = inputs[0, ...]  # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)


