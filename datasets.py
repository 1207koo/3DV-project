import os
from dataset.coco import *
from dataset.transform import *
from args import *

def get_dataset(mode='train'):
    if args.dataset == 'COCO2014':
        return COCODataset(os.path.join(args.dataset_path, args.dataset), use=['%s2014'%mode], min_size=(args.img_size, args.img_size), transform=[RandomCrop(args.img_size), ColorNormal()])
    else:
        raise NotImplementedError