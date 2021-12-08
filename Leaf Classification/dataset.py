# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/8 上午10:24
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image
data_root = r'leaf-classification/images/'

class leaf_Dataset(data.Dataset):
    def __init__(self,is_train=True,transform=None):
        self.is_train = is_train
        self.transform = transform
        self.images = []
        self.labels = []
        if is_train:
            file = open('train.txt','r')
            lines = file.readlines()
            for line in lines:
                res = line[:-1]
                image = res.split(' ')[0]
                label = int(res.split(' ')[1])
                self.images.append(image)
                self.labels.append(label)
            print(self.images)
            print(self.labels)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_name = self.images[index] + '.jpg'
        image_path = data_root + image_name
        img = Image.open(image_path).convert('RGB')
        # print(img)
        img = self.transform(img)
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))
        return img, label

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
