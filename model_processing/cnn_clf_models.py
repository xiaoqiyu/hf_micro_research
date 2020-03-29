#!/user/bin/env python
# coding=utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: ml_clf_models.py
@time: 20-02-23 下午4:03
@desc:
'''

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt

# ref: https://morvanzhou.github.io/tutorials/machine-learning/torch/4-01-CNN/
# ref:https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)  # net architecture
