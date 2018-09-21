import cv2, random, os
import h5py
from mxnet import nd, autograd, gluon, image
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

class ArtLoader(gluon.data.Dataset):
    def __init__(self, folder, final_resolution):

        self.folder = folder
        self.final_resolution = final_resolution
        self.images = [x for x in os.listdir(self.folder) if x.endswith('.png') or x.endswith('.jpg')]
        self.len = len(self.images)

    def __getitem__(self, item):
        item = random.sample(self.images, 1)[0]
        img = cv2.cvtColor(cv2.imread(self.folder+item),cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        if h < self.final_resolution:
            img = cv2.resize(img, (self.final_resolution, int(h/w*self.final_resolution)), interpolation=cv2.INTER_CUBIC)
        if w < self.final_resolution:
            img = cv2.resize(img, (int(w/ h * self.final_resolution), self.final_resolution), interpolation=cv2.INTER_CUBIC)
        size = (self.final_resolution, int(w/h*self.final_resolution)) if h<w \
            else (int(h/w*self.final_resolution), self.final_resolution)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        h, w, _ = img.shape
        if h<w:
            x = random.randint(0, w - self.final_resolution)
            img = img[:, x:x+self.final_resolution,:]
        else:
            x = random.randint(0, h - self.final_resolution)
            img = img[x:x + self.final_resolution,:,:]
        if random.random()<0.5:
            img = img[:,::-1,:]

        img = (np.transpose(img, axes=(2,0,1)).astype(np.float32) / 127.5) - 1
        return img

    def __len__(self):
        return self.len*10000