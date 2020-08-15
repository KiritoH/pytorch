# -*- coding:utf-8 -*-
"""
@file name  : tensorboard_methods.py
# @author     : TingsongYu https://github.com/TingsongYu
@date       : 2019-10-24
@brief      : tensorboard方法使用（一）scalars and histogram
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

# ----------------------------------- 0 SummaryWriter -----------------------------------
flag = 0
# flag = 1
if flag:

    log_dir = "./train_log/test_log_dir"
    # writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")
    writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")

    for x in range(100):
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.close()


# ----------------------------------- 1 scalar and scalars -----------------------------------
flag = 0
# flag = 1
if flag:

    max_epoch = 100

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(max_epoch):

        # add_scalar是写单条曲线,而scalars可以写多条曲线
        # 第一个属性是标签,第二个属性是y的值,而第三个是x轴
        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                 "xcosx": x * np.cos(x)}, x)

    writer.close()

# ----------------------------------- 2 histogram -----------------------------------
# 统计直方图与多分位数折线图
# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    # 画两次
    for x in range(2):

        np.random.seed(x)
        # 0-99,100个数据点
        data_union = np.arange(100)
        # 1000个随机符合正态分布的数据点
        data_normal = np.random.normal(size=1000)

        writer.add_histogram('distribution union', data_union, x)
        writer.add_histogram('distribution normal', data_normal, x)

        plt.subplot(121).hist(data_union, label="union")
        plt.subplot(122).hist(data_normal, label="normal")
        plt.legend()
        plt.show()

    writer.close()
