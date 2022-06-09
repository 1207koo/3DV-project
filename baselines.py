import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from d2_net.lib.model_test import D2Net
from d2_net.lib.pyramid import process_multiscale
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from args import *
from util import *

_d2net = None
def extract_d2net(batch, model=None):
    b = batch.shape[0]
    if model is None and _d2net is None:
        _d2net = D2Net(model_file='./d2_net/models/d2_tf.pth', use_relu=True, use_cuda=True)
    kpts, descs = [], []
    if model is None:
        for i in range(b):
            kpt, _, des = process_multiscale(batch[i:i+1], _d2net, scales=[1])
            kpts.append(kpt[:, :2])
            descs.append(des)
    else:
        for i in range(b):
            kpt, _, des = process_multiscale(batch[i:i+1], model, scales=[1])
            kpts.append(kpt[:, :2])
            descs.append(des)
    return kpts, descs

sift = None
def extract_sift(batch, model=None):
    if sift is None:
        sift = cv2.xfeatures2d.SIFT_create()
    kpts, dess = [], []
    for i in batch:
        k, d = sift.detectAndCompute(i, None)
        kpts.append(k)
        dess.append(d)
    return kpts, dess

orb = None
def extract_orb(batch, model=None):
    if orb is None:
        orb = cv2.ORB_create()
    kpts, dess = [], []
    for i in batch:
        k, d = orb.detectAndCompute(i, None)
        kpts.append(k)
        dess.append(d)
    return kpts, dess

_superpoint = None
def extract_superpoint(batch, model=None):
    if model is None and _superpoint is None:
        _superpoint = SuperPointFrontend('./SuperPointPretrainedNetwork/superpoint_v1.pth', 4, 0.015, 0.7, True)
    if model is None:
        pts, desc, heatmap = _superpoint.run(batch)
    else:
        pts, desc, heatmap = model.run(batch)
    return pts, desc


def extract_teacher(batch, model=None):
    with torch.no_grad():
        kpt, desc = eval('extract_%s(batch, model)'%args.teacher)
        if type(kpt[0]) != torch.Tensor:
            kpt = [torch.from_numpy(np.array(kpt_)[:, ::-1].copy()).to(args.device) for kpt_ in kpt]
        if type(desc[0]) != torch.Tensor:
            desc = [torch.from_numpy(np.array(desc_)).to(args.device) for desc_ in desc]
        if args.l2:
            desc = [F.normalize(desc_) for desc_ in desc]
        return kpt, desc