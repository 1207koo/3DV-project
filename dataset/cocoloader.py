import os
import random
import tqdm
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class COCODataset(Dataset):
    def __init__(self, dataset_path='/gallery_moma/junseo.koo/dataset/COCO2014', use=['train2014'], transform=[], warp_transform=[]):
        assert os.path.isdir(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.use = use
        self.transform = transform
        self.warp_transform = warp_transform

        self.files = []
        for split in self.use:
            self.files += glob.glob(os.path.join(self.dataset_path, split, '*.jpg'))
        self.files = sorted(self.files)
    
    def toTensor(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = cv2.imread(filename)
        if len(self.transform) > 0:
            for tf in self.transform:
                img = tf(img)
        if len(self.warp_transform) > 0:
            img_warp = img
            for tf in self.warp_transform:
                img_warp = tf(img_warp)
            return self.toTensor(img), self.toTensor(img_warp)
        return self.toTensor(img)

