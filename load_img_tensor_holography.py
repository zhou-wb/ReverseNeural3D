import os

import numpy

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2 as cv
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms

import numpy as np
# from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
# import imageio
import sys
import seaborn as sns
import matplotlib.pyplot as plt

class Tensor_Holography_dataset(object):
    def __init__(self, dir, color_transform=None, depth_transform=None, channel=None):
        self.dir = dir
        self.color_transform = color_transform
        self.depth_transform = depth_transform
        self.channel = channel
        self.amp_dir = self.dir + "/amp"
        self.depth_dir = self.dir + "/depth"
        self.img_dir = self.dir + "/img"
        self.phs_dir = self.dir + "/phs"

        self.amp_lst = os.listdir(self.amp_dir)
        self.depth_lst = os.listdir(self.depth_dir)
        self.img_lst = os.listdir(self.img_dir)
        self.phs_lst = os.listdir(self.phs_dir)

    def __len__(self):
        return len(self.amp_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_lst[index])
        depth_path = os.path.join(self.depth_dir, self.depth_lst[index])
        # print(img_path)
        # print(depth_path)
        original_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        depth_img = cv.imread(depth_path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        depth_img = depth_img[:, :, 0]

        masks = self.cal_masks(depth_img)
        masks = numpy.array(masks)

        trans = transforms.ToTensor()
        if type(original_img) != torch.Tensor:
            original_img = trans(original_img)
        if type(masks) != torch.Tensor:
            masks_tensor = torch.tensor(masks)

        img_id = self.img_lst[index]
        green_channel = original_img[1, :, :].unsqueeze(0)

        return [green_channel, masks_tensor, img_id]
    def cal_masks(self, depth_img):
        planes = [0, 0.33, 0.66, 1.0]
        masks = []

        for i in range(1, len(planes)):
            low = planes[i - 1]
            high = planes[i]
            mask = ((depth_img >= low) & (depth_img < high)).astype(int)
            masks.append(mask)
        return masks

# test = Tensor_Holography_dataset("/Users/lihanshuo/hku/sem2/project/code/ReverseNeural3D-back_forward_split/test_192")
# couter = 0
# for c in test:
#     img = c[0]
#     masks = c[1]
#     img_id = c[2]
#     print(img.shape)
#     print(masks.shape)
#     print(img_id)
#     couter += 1
# print(couter)


# for item in my_img_loader:
#     print(item)
#     0 -- img #1 * w * h tensor
#     1 -- masks 0 1 2 # 3 * w * h tensor
#     2 -- img_id # str



# img = cv.imread("/Users/lihanshuo/hku/sem2/project/code/ReverseNeural3D-back_forward_split/test_192/img/3965.exr",
#           cv.IMREAD_UNCHANGED)
# cv.imshow("test.exr", img)
# cv.waitKey(0)
# print(img[:, :, 0])
