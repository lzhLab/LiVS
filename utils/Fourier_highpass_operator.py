
import cv2
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
class HFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=5,
                 use_cuda=False):
        super(HFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'
        la_2D = torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 1.0, 2.0, 1.0, 0.0],
                              [1.0, 2.0, -15, 2.0, 1.0],
                              [0.0, 1.0, 2.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0],
                              ])

        self.h_filter = nn.Conv2d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=k_sobel,
                                  padding=k_sobel // 2,
                                  bias=False)
        self.h_filter.weight[:] = la_2D


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        for c in range(C):
            grad_x = grad_x + self.h_filter(img[:, c:c + 1])
        grad_x = grad_x / C
        # grad_x[grad_x<0] = 0
        grad_x = torch.abs(grad_x)
        grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min())
        return grad_x

def convolve(image):
    height,width = image.shape
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

def _down_scales(x, sc):
    _, _, h, w = x.size()
    return F.interpolate(x, size=(h//sc, w//sc), mode='bilinear', align_corners=True)
