import random

import numpy as np
import torch
import torchvision
from PIL import Image
from numpy import *
from torch.utils.data import Dataset
from torchvision import transforms

from utils.mytransforms import mytransforms


class dataload(Dataset):
    def __init__(self, path, H, W, pow_n=8, aug=False, phase='valid'):
        self.path = path
        self.H = H
        self.W = W
        self.pow_n = pow_n
        self.aug = aug
        self.phase = phase

        if self.phase == 'predict':
            self.data_num = 1

        else:
            self.dinfo = torchvision.datasets.ImageFolder(root=self.path)
            self.mask_num = int(len(self.dinfo.classes))
            self.data_num = int(len(self.dinfo.targets) / self.mask_num)
            self.path_mtx = np.array(self.dinfo.samples)[:, :1].reshape(self.mask_num, self.data_num)

        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(),
                                              mytransforms.Affine(0,
                                                                  translate=[0, 0],
                                                                  scale=1,
                                                                  fillcolor=0),
                                              transforms.ToTensor()])
        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random())])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.phase == 'predict':
            input = Image.open(self.path)
            input = self.mask_trans(input)
            input = self.norm(input)

            return input

        else:
            imgs = torch.empty(self.mask_num, self.H, self.W, dtype=torch.float)

            if self.aug:
                self.mask_trans.transforms[2].degrees = random.randrange(-25, 25)
                self.mask_trans.transforms[2].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
                self.mask_trans.transforms[2].scale = random.uniform(0.9, 1.1)

            for k in range(self.mask_num):
                X = Image.open(self.path_mtx[k, idx])
                if k == 0 and self.aug:
                    X = self.col_trans(X)
                imgs[k] = self.mask_trans(X)

            input, heat = self.norm(imgs[0:1]), imgs[1:38]
            heat = torch.pow(heat, self.pow_n)
            heat = heat / heat.max()

            return [input, heat]
