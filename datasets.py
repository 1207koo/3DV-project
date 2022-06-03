import os
from dataset.coco import *
from dataset.transform import *
from args import *

def get_dataset(mode='train'):
    if args.dataset == 'COCO2014':
        return COCODataset(os.path.join(args.dataset_path, args.dataset), use=['%s2014'%mode], transform=[StrongCrop(int(1.5 * args.img_size)), RandomCrop(args.img_size), ColorNormal()])
    else:
        raise NotImplementedError