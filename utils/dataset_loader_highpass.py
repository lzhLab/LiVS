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
from square_wave import creat_gabor 

def LoaderNames(names_path):
    f = open(names_path, 'r')
    names = f.read().splitlines()
    f.close()
    return names


def convolve(image):
    height,width = image.shape
    #Laplace Operator
    filter = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 1.0, 2.0, 1.0, 0.0],
                          [1.0, 2.0, -15, 2.0, 1.0],
                          [0.0, 1.0, 2.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          ])
    h,w = filter.shape
    new_h = height-h+1
    new_w = width-h+1
    new_img = np.zeros((new_h,new_w),dtype=np.float)
    for i in range(new_h):
        for j in range(new_w):
            new_img[i,j]=np.sum(image[i:i+h,j:j+w]*filter)
    new_img = new_img.clip(0,255)
    return new_img


def default_loader(path, model_type):
    image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)#cv2.imread(path)
    x,y = active_block(image)
    image = image[x:x+256,y:y+256]/(np.max(image)+1)
    gabor = creat_gabor(path.replace(str(model_type),'gabor_train'))
    # gaussianblur
    gabor_gs = cv2.GaussianBlur(gabor, (9, 9), 0)
    # filter
    hp = convolve(gabor_gs) 
    hp = hp[x:x + 256, y:y + 256]
    hp = hp/(np.max(hp)+1)
    hp.resize(256,256)
    
    gabor = gabor[x:x + 256, y:y + 256]
    gabor_w = gabor/(np.max(gabor)+1)
    label = cv2.imdecode(np.fromfile(path.replace(str(model_type),'trainmask'), dtype=np.uint8), -1)
    label = label[x:x + 256, y:y + 256] / 255.

    return image,gabor_w,hp,label

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
