import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter

import os
import cv2
import math
from tqdm import tqdm


def show_heatmap(tgt, num=0, show=True, save=None):
    tgt = tgt
    plt.imshow(tgt, cmap=plt.cm.jet)
    plt.xlabel(str(num))
    if show:
        plt.show()
    if save != None:
        plt.savefig(save)


class perspective:
    def __init__(self, x1, y1, siz1, x2, y2, siz2):
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.siz1=siz1
        self.siz2=siz2

    def get(self, x, y):
        # return the kernal size and sigma of (x,y)
        rtn = self.cal(x, y)
        return (rtn*2+1), rtn//3
    
    def cal(self, x, y):
        t = math.ceil((y-self.y1)/(self.y2-self.y1)*self.siz2 -
                      (y-self.y2)/(self.y2-self.y1)*self.siz1)
        t = max(12, t)
        return t


def gaussian_filter_density(gt, pts, mat2):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)
    P = perspective(mat2[0], mat2[1], mat2[2], mat2[3], mat2[4], mat2[5])

    if gt_count == 0:
        return density

    #     print('=======generating ground truth=========')
    Fixed_H = np.multiply(cv2.getGaussianKernel(15, 4), (cv2.getGaussianKernel(15, 4)).T)
    print(Fixed_H.sum())
    H = Fixed_H
    h, w = gt.shape

    # print('imageshape: ', gt.shape)

    for i, point in enumerate(pts):
        x = min(w, max(0, abs(int(point[0]))))  # read x?
        y = min(h, max(0, abs(int(point[1]))))  # read y?
        # pixel: (y,x)

        # get new size of gaussian kernel by perspective information
        r, sigma = P.get(x, y)
        c = r

        if x >= w or y >= h:
            continue
        x1 = x - int(c / 2)
        x2 = x + int(c / 2)
        y1 = y - int(r / 2)
        y2 = y + int(r / 2)

        dfx1 = 0
        dfx2 = 0
        dfy1 = 0
        dfy2 = 0
        change_H = True
        if x1 <= 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        if y1 <= 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        if x2 >= w:
            dfx2 = x2 - (w - 1)
            x2 = w - 1
            change_H = True
        if y2 >= h:
            dfy2 = y2 - (h - 1)
            y2 = h - 1
            change_H = True

        x1h = dfx1
        y1h = dfy1
        x2h = c - 1 - dfx2
        y2h = r - 1 - dfy2

        if change_H:
            H = np.multiply(cv2.getGaussianKernel(y2h - y1h + 1, sigma),
                            (cv2.getGaussianKernel(x2h - x1h + 1, sigma)).T)

        # print(H.sum())
        density[y1:y2 + 1, x1:x2 + 1] += H

        if change_H:
            H = Fixed_H

    #     print('===========done=============')
    return density


def main(image_root_path, image_gt_path, image_gt_ex_path, image_save_path, image_gt_save_path):

    images = os.listdir(image_root_path)
    for imagename in images:
        print('current processing: ', imagename)
        r = c = 15
        sigma = 4
        if imagename.find('.jpg')<=0:
            continue
        imagepath = os.path.join(image_root_path, imagename)
        gtpath = os.path.join(image_gt_path, imagename).replace('.jpg','.mat').replace('IMG','GT_IMG')
        gtexpath = os.path.join(image_gt_ex_path, imagename).replace('.jpg','.npy').replace('IMG','GT_EX_IMG')
        
        imagesavepath = os.path.join(image_save_path, imagename)
        imagegtsavepath = os.path.join(image_gt_save_path, imagename)

        img = cv2.imread(imagepath)
        mat = scipy.io.loadmat(gtpath)
        mat2 = np.load(gtexpath)
        points = mat['image_info'][0][0][0][0][0]

        density = np.zeros((img.shape[0], img.shape[1]))
        density1 = gaussian_filter_density(density, points, mat2)

        #show_heatmap(density1)



        #return

        cv2.imwrite(imagesavepath, img)
        np.save(imagegtsavepath.replace('.jpg', '.npy'), density1)
        print('total actual number: ', len(points))
        print('predict estimates: ', density1.sum())


    print('==>saving finish.')


if __name__ == '__main__':
    loc = 'C:/Users/ooo69'

    image_root_path = loc + '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images'
    image_gt_path = loc + '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth'
    image_gt_ex_path = loc + '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth_ex'
    image_save_path = loc + '/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Train/'
    image_gt_save_path = loc + '/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Train_gt'

    main(image_root_path, image_gt_path, image_gt_ex_path, image_save_path, image_gt_save_path)

    image_root_path = loc + '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images'
    image_gt_path = loc + '/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/ground_truth'
    image_save_path = loc + '/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Test/'
    image_gt_save_path = loc + '/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Test_gt'

    # main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)
