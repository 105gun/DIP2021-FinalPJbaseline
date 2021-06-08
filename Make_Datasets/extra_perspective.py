import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tkinter as tk

import numpy as np
import scipy
import scipy.io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter

import os
import cv2
import math
from tqdm import tqdm

def col(img,x,y,h,w,siz=10):
    img_=img.copy()
    x1=max(0,x-siz)
    x2=min(w,x+siz)
    y1=max(0,y-siz)
    y2=min(h,y+siz)
    for i in range(x1,x2):
        for j in range(y1,y2):
            img_[j][i] = ([0, 255, 0]+img_[j][i])/2
    return img_

def get(img, points, total, crt):
    img = img[:, :, ::-1]
    h,w=img.shape[0], img.shape[1]
    while True:
        print('\n-------------\nTotal:', total)
        print('Current:', crt)
        x = math.ceil(points[crt][0])
        y = math.ceil(points[crt][1])
        print('x:',x,"y:",y)
        img_=col(img,x,y,h,w,5)
        plt.imshow(img_)
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
        rtn=int(input('Input size:'))
        if(rtn<0):
            crt=-rtn
        else:
            img_ = col(img, x, y, h, w, rtn)
            plt.imshow(img_)
            plt.get_current_fig_manager().window.state('zoomed')
            plt.show()
            rtn_=input('Are you sure?')
            if rtn_=='y':
                return points[crt][0], points[crt][1], rtn

def main(image_root_path, image_gt_path, image_gt_ex_path):

    images = os.listdir(image_root_path)
    for imagename in images:
        print('current processing: ', imagename)
        if imagename.find('.jpg') <= 0:
            continue
        imagepath = os.path.join(image_root_path, imagename)
        gtpath = os.path.join(image_gt_path, imagename).replace(
            '.jpg', '.mat').replace('IMG', 'GT_IMG')
        gtexpath = os.path.join(image_gt_ex_path, imagename).replace(
            '.jpg', '.npy').replace('IMG', 'GT_EX_IMG')
        # imagesavepath = os.path.join(image_save_path, imagename)
        # imagegtsavepath = os.path.join(image_gt_save_path, imagename)

        img = cv2.imread(imagepath)
        mat = scipy.io.loadmat(gtpath)
        mat2 = np.zeros(6)
        points = mat['image_info'][0][0][0][0][0]
        points = points[np.argsort(points[:, 1])]
        total = len(points)

        print(type(points),points)

        mat2[0], mat2[1], mat2[2] = get(img, points, total, total//5)
        mat2[3], mat2[4], mat2[5] = get(img, points, total, total-1-total//100)


        # density = np.zeros((img.shape[0], img.shape[1]))
        # density1 = gaussian_filter_density(density, points, r, c, sigma)

        # show_heatmap(density1)
        print(mat2)
        np.save(gtexpath, mat2)

    print('==>saving finish.')


if __name__ == '__main__':
    loc = 'C:/Users/ooo69'

    image_root_path = loc + \
        '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images'
    image_gt_path = loc + \
        '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth'
    image_gt_ex_path = loc + \
        '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth_ex'

    main(image_root_path, image_gt_path, image_gt_ex_path)
