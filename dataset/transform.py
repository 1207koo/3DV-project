import os
import random
import cv2
import numpy as np
import torch

class Resize:
    def __init__(self, img_size):
        if type(img_size) == int:
            self.h = img_size
            self.w = img_size
        else:
            self.h = img_size[0]
            self.w = img_size[1]
    def __call__(self, img):
        return cv2.imresize(img, (self.w, self.h))

class ResizeRatio:
    def __init__(self, ratio):
        if type(ratio) == float:
            self.hr = ratio
            self.wr = ratio
        else:
            self.hr = ratio[0]
            self.wr = ratio[1]
    def __call__(self, img):
        h, w = img.shape[:2]
        return cv2.imresize(img, (int(w * self.wr), int(h * self.hr)))

class RandomCrop:
    def __init__(self, img_size):
        if type(img_size) == int:
            self.h = img_size
            self.w = img_size
        else:
            self.h = img_size[0]
            self.w = img_size[1]
    def __call__(self, img):
        h, w = img.shape[:2]
        hp = np.random.randint(h - self.h + 1)
        wp = np.random.randint(w - self.w + 1)
        return img[hp:hp+self.h, wp:wp+self.w]

class StrongCrop:
    def __init__(self, img_size):
        if type(img_size) == int:
            self.h = img_size
            self.w = img_size
        else:
            self.h = img_size[0]
            self.w = img_size[1]
    def __call__(self, img):
        h, w = img.shape[:2]
        hs, ws = self.h, self.w
        if hs > h:
            ws = int(ws * h / hs)
            hs = h
        if ws > w:
            hs = int(hs * w / ws)
            ws = w
        hp = np.random.randint(h - hs + 1)
        wp = np.random.randint(w - ws + 1)
        cimg = img[hp:hp+hs, wp:wp+ws]
        return cv2.resize(cimg, (self.w, self.h))

class ColorNormal:
    def __init__(self, color_max=255, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
        self.cmax = color_max
        self.mean = mean
        self.std = std
    def __call__(self, img):
        return (img / self.cmax - self.mean.reshape((1, 1, 3))) / self.std.reshape((1, 1, 3))

class Homography:
    def __init__(self, h):
        self.h = h
    def __call__(self, img):
        raise NotImplementedError
        # create random 3*3 homography
        homography = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # warp(image)
        return img, homography