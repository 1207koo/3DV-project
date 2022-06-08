import os
import random
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm


class COCODataset(Dataset):
    def __init__(self, dataset_path='', use=['train2014'], min_size=(256, 256), transform=[], warp_transform=[]):
        assert os.path.isdir(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.use = use
        self.min_size = min_size
        self.transform = transform
        self.warp_transform = warp_transform

        self.files = []
        for split in self.use:
            if self.min_size is None:
                self.files += sorted(glob.glob(os.path.join(self.dataset_path, split, '*.jpg')))
            else:
                min_file = 'dataset/%s_%d_%d.txt'%(split, min_size[0], min_size[1])
                if not os.path.isfile(min_file):
                    imgs = []
                    for img_path in tqdm(sorted(glob.glob(os.path.join(self.dataset_path, split, '*.jpg'))), desc='filtering images'):
                        img = cv2.imread(img_path)
                        if img.shape[0] >= self.min_size[0] and img.shape[1] >= self.min_size[1]:
                            imgs.append(img_path.split('/')[-1])
                    with open(min_file, 'w') as f:
                        for img_path in imgs:
                            f.write(img_path + '\n')
                with open(min_file, 'r') as f:
                    lines = f.readlines()
                for i in range(len(lines)):
                    lines[i] = os.path.join(self.dataset_path, split, lines[i].replace('\n', ''))
                self.files += lines
        self.files = sorted(self.files)
    
    def toTensor(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32))

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
            homography = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            for tf in self.warp_transform:
                img_warp, h = tf(img_warp)
                homography = h @ homography
            return self.toTensor(img), self.toTensor(img_warp), homography
        return self.toTensor(img)

