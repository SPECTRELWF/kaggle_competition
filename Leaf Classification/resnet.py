# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/8 上午10:24
import torch
import torchvision.models
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

class resnet101(nn.Module):
    def __init__(self, num_classes=1000):
        super(resnet101, self).__init__()
        self.num_classes = num_classes
        self.feature_extract = torchvision.models.resnet101(pretrained=True)
        self.net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.net(x)
        return x


# x = torch.randn((1,3,224,224))
# net = resnet101(num_classes=99)
# print(net)
# print(net(x).shape)
