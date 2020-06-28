# -*- coding:utf-8 -*-


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2


def my_random_crop(image):
    print(image.shape)
    print(image.shape[2])
    ret = []
    for i in range(image.shape[2]):
        print("hhh", i)
        y = int(np.random.randint(1080 - 512 + 1))
        x = int(np.random.randint(1920 - 512 + 1))
        h = 512
        w = 512
        image_crop = image[y:y + h, x:x + w, i]
        # image_crop = image[x:x + w, y:y + h, i]
        print("image_crop.shpe:", image_crop.shape)
        if image_crop.shape[0] != 512 or image_crop.shape[1] != 512:
            print('image size error')
        ret.append(image_crop)
    img = np.array(ret)
    print("img的shape:", img.shape)
    # return img
    return img.transpose((1,2,0))


def test_transe():
    A = np.array([[[0, 1, 2, 3],
                   [4, 5, 6, 7]],

                  [[8, 9, 10, 11],
                   [12, 13, 14, 15]]])
    print(A.shape)

    print(A.transpose((2, 0, 1)))
    print(A.transpose((2, 0, 1)).shape)


if __name__ == '__main__':
    path = "train_image/0/WIN_20200624_14_06_06_Pro.jpg"
    imc = cv2.imread(path)
    print(imc.shape)
    # imc = imc.transpose((2,0,1))
    # print(imc.shape)
    # print(imc.shape)

    img = my_random_crop(imc)
    print(img.shape)

    # print(imc[:,:,2].shape)

