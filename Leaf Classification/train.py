# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/8 上午10:25


"""
使用imagenet预训练的rennet101来在树叶数据集上面进行微调
"""
import torch
import torchvision.transforms as transforms
from dataset import leaf_Dataset
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from resnet import resnet101
#使用Adam优化器来训练网络，不冻结参数

# 设置hyperparameter

epoch = 200
lr = 1e-3
b1 = 0.9
b2 = 0.999
device = torch.device('cuda:0')
train_loss = []
# 初始化网络模型
net = resnet101(num_classes=99)
net.to(device)

# load data
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])
data = leaf_Dataset(is_train=True,transform=transforms)
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=64,
                                         shuffle=True)

loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr=lr,betas=(b1,b2))


for epoch in range(1,epoch + 1):
    for i, (x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = net(x)
        loss = loss_func(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        print("epoch: %d   batch_idx:%d   loss:%.3f" %(epoch,i,loss.item()))
    torch.save(net.state_dict(),'model/epoch:%d'%epoch + '.pth')
from utils import plot_curve
plot_curve(train_loss)