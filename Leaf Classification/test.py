# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/8 下午5:42
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from resnet import resnet101
import torch.nn.functional as F

image_path = r'leaf-classification/images'
f = open('test.txt','r')
tmp = f.readlines()
test_file = []
for i in tmp:
    i = i[:-1]
    test_file.append(i+'.jpg')
print(test_file)

device = torch.device('cuda:0')
net = resnet101(num_classes=99)
print('load weight........')
net.load_state_dict(torch.load('model/epoch:200.pth'))
net.to(device)
net.eval()
transformss = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
res = []
with torch.no_grad():
    for image in test_file:
        img = Image.open(os.path.join(image_path,image)).convert('RGB')
        img = transformss(img)
        img = torch.unsqueeze(img,dim=0)
        img = img.to(device)
        # print(img.shape)
        pred = net(img)
        pred = F.softmax(pred).flatten()
        pred = pred.cpu().numpy()
        print(pred)
        res.append(pred)
        np.savetxt("result.csv",res,delimiter = ',')
