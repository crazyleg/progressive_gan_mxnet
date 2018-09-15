import cv2, random, os
import h5py
from mxnet import nd, autograd, gluon, image
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

class LowISOSet(gluon.data.Dataset):
    def __init__(self, folder, final_resolution):

        self.folder = folder
        self.final_resolution = final_resolution
        self.images = [x for x in os.listdir(self.folder) if x.endswith('.png')]
        self.len = len(self.images)

    def __getitem__(self, item):
        item = random.sample(self.images_low_iso, 1)[0]
        img = cv2.cvtColor(cv2.imread(self.folder_low_iso+item),cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        size = (self.final_resolution, int(w*(h/self.final_resolution))) if h>w \
            else (int(h*(w/self.final_resolution), self.final_resolution))
        img = cv2.resize(img, size)
        if h>w:
            x = random.randint(0,h-self.final_resolution-1)
            img = img[x:x+512,:,:]
        else:
            x = random.randint(0, h - self.final_resolution - 1)
            img = img[:, x:x + 512, :]
        img = np.transpose(img, axes=(2,0,1)).astype(np.float32)
        return img

    def __len__(self):
        return self.len