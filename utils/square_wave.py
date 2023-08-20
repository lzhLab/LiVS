from PIL import Image, ImageOps
import numpy as np
import glob
import cv2
import os
import math

def bi_demo(image):
    dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15) 
    img_medianBlur = cv2.medianBlur(dst, 3) 
    return img_medianBlur

def gabor(sigma, theta, Lambda, gamma, ksize, cos_or_sin):
    """cos_or_sin=1 return cos square_wave."""
    """cos_or_sin=0 return sin square_wave."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    xmax = int(ksize[0]/2)
    ymax = int(ksize[1]/2)
    xmax = np.ceil(max(1, xmax))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor_triangle = np.zeros([ksize[0], ksize[1]])
#    gabor_triangle = np.zeros([2*ksize[0]+1, 2*ksize[1]+1])
    if cos_or_sin == 1:
        for j in range(0, 100, 1):
            gabor_triangle_tmp = 4 / math.pi * (1 / (2 * j + 1)) * np.cos(2 * np.pi / Lambda * ((2 * j + 1)) * x_theta + j * math.pi )
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    if cos_or_sin == 0:
        for j in range(0, 100, 1):
            gabor_triangle_tmp = 4 / math.pi * (1 / (2 * j + 1)) * np.cos(2 * np.pi / Lambda * ((2 * j + 1)) * x_theta + math.pi/2)
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * gabor_triangle
    return gb


def gabor_iteration(image, count):

    sigm = 3
    lambd = 2*sigm
    gamm = 1
    # k_size = (sigm,sigm)
    k_size = (5, 5)
    for time in range(count):
        H, W = image.shape
        out = np.zeros([H, W], dtype=np.float32)
        direction = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]

        for i, t in enumerate(direction):
            # gabor_kernel = cv2.getGaborKernel(ksize=(5, 5), sigma=5, theta=t * math.pi / 180 * 2, lambd=10, gamma=1.2,
            #                                   psi=0)
            gabor_kernel = gabor(sigma = sigm, theta=t * math.pi / 180 * 2, Lambda=lambd, gamma=gamm,ksize=k_size,cos_or_sin=1)
            new_gabor_kernel = np.where(gabor_kernel > 0, gabor_kernel, 0)

            count1 = np.sum(new_gabor_kernel)
            gabor_kernel = gabor_kernel / count1

            result = cv2.filter2D(image, -1, gabor_kernel)
            out += result

        # plt.show()
        out = out / len(direction)
        # out = out.astype(np.uint8)
        result1 = out

        out = np.zeros([H, W], dtype=np.float32)
        direction = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        for i, t in enumerate(direction):
            # gabor_kernel = cv2.getGaborKernel(ksize=(5, 5), sigma=5, theta=t * math.pi / 180 * 2, lambd=10, gamma=1.2,
            #                                   psi=0)
            gabor_kernel = gabor(sigma = sigm, theta=t * math.pi / 180 * 2, Lambda=lambd, gamma=gamm,ksize=k_size,cos_or_sin=0)
            new_gabor_kernel = np.where(gabor_kernel > 0, gabor_kernel, 0)
            count1 = np.sum(new_gabor_kernel)
            gabor_kernel = gabor_kernel / count1
            result = cv2.filter2D(image, -1, gabor_kernel)
            out += result
        out = out / len(direction)
        # out = out.astype(np.uint8)
        result2 = out

        gabor_image = result1 + result2
        image = gabor_image
    return gabor_image


def creat_gabor(image):
    image = image - np.sum(image) / (np.sum(image > 0)+1)
    gabor_result = gabor_iteration(image, 1)
    gabor_result = np.clip(gabor_result, 1, 255)
    return gabor_result
