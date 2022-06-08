import cv2
from d2_net.lib.model import D2Net
from d2_net.lib.pyramid import process_multiscale
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from args import *

_d2net = None
def extract_d2net(batch, model=None):
    if model is None and _d2net is None:
        _d2net = D2Net(model_file='./d2_net/models/d2_tf.pth', use_cuda=True)
    if model is None:
        kpt, _, des = process_multiscale(batch, _d2net, scales=[1])
    else:
        kpt, _, des = process_multiscale(batch, model, scales=[1])
    return kpt, des

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
    kpt, desc = eval('extract_%s(batch, model)'%args.teacher)
    if type(kpt) == list:
        kpt = torch.from_numpy(np.array(kpt)[:, ::-1]).to(args.device)
    if type(desc) == list:
        desc = torch.from_numpy(np.array(desc)).to(args.device)
    return kpt, desc