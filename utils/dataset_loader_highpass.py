# -*- coding:utf-8 -*-

"""
@author:gz
@file:dataset_loader.py
@time:2022/5/318:20
"""
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import nibabel as nib
import os
import glob
import cv2
from .square_wave import creat_gabor
from PIL import Image
import random
import scipy.signal

def LoaderNames(names_path):
    f = open(names_path, 'r')
    names = f.read().splitlines()
    f.close()
    return names


def convolve(image):

    #Laplace Operator
    filter = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 1.0, 2.0, 1.0, 0.0],
                          [1.0, 2.0, -15, 2.0, 1.0],
                          [0.0, 1.0, 2.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          ])
    new_img = scipy.signal.convolve2d(image, filter, mode='same')
    new_img = new_img.clip(0,255)
    return new_img

def default_loader(path, model_type):
    ima_size = 512
    image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)#cv2.imread(path)

    height, width = image.shape

    padded_img = np.zeros((ima_size, ima_size), dtype=np.uint8)
    padded_lab = np.zeros((ima_size, ima_size), dtype=np.uint8)
    # scale = 512.0 / max(height, width)
    # image = cv2.resize(image, None, fx=scale, fy=scale)

    y_start = (ima_size - height) // 2
    x_start = (ima_size - width) // 2
    padded_img[y_start:y_start + height, x_start:x_start + width] = image

    gabor = creat_gabor(padded_img)
    gabor_gs = cv2.GaussianBlur(gabor, (9, 9), 0)
    hp = convolve(gabor_gs)

    # st = str(random.randint(0, 9))
    # Image.fromarray(gabor).convert('L').save(os.path.join("dataset", st+"_gb.png"))
    # Image.fromarray(hp).convert('L').save(os.path.join("dataset", st + "_hp.png"))

    gabor_w = gabor / (np.max(gabor) + 1)
    hp = hp / (np.max(hp) + 1)
    padded_img = padded_img/(np.max(image)+1)

    label = cv2.imdecode(np.fromfile(path.replace(str(model_type),'trainmask'), dtype=np.uint8), -1)
    padded_lab[y_start:y_start + height, x_start:x_start + width] = label

    return padded_img,gabor_w,hp,padded_lab

#cut to 256*256
def active_block(img):
    ret, thresh = cv2.threshold(img, 0, 255, 0)
    contours,hier = cv2.findContours(thresh, 1, 2)
    if len(contours)==0: 
        return 0,0    
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    xx = x + w // 2 - 128
    yy = y + h // 2 - 128
    if xx < 0 or xx > 255: xx = 0
    if yy < 0 or yy > 255: yy = 0
    return yy,xx

class MyDataset(Dataset): 
    def __init__(self, model_type, data_filename,sub_name='', transform=None, loader=default_loader): 
        super(MyDataset, self).__init__()  
        imgs = glob.glob(os.path.join(data_filename,model_type+'/'+sub_name+'*'))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.data_filename = data_filename
        self.typestr = model_type
    def __getitem__(self, index):  
        img_str = self.imgs[index] 
        img,gb,hp,label = self.loader(img_str,self.typestr)
        if self.transform is not None:
            img = self.transform(img)  
            label = self.transform(label)  
            gb = self.transform(gb)
            hp = self.transform(hp)
        return img, gb, hp, label

    def __len__(self):  
        return len(self.imgs)
