import cv2
from d2_net.lib.model_test import D2Net
from d2_net.lib.pyramid import process_multiscale
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend

_d2net = D2Net(model_file='/gallery_tate/jinseo.jeong/3dcv/models/d2_tf.pth',
               use_relu=True, use_cuda=True)
def extract_d2net(batch):
    kpt, _, des = process_multiscale(batch, _d2net, scales=[1])
    return kpt, des

sift = cv2.xfeatures2d.SIFT_create()
def extract_sift(batch):
    kpts, dess = [], []
    for i in batch:
        k, d = sift.detectAndCompute(i, None)
        kpts.append(k)
        dess.append(d)
    return kpts, dess

orb = cv2.xfeatures2d.ORB_create()
def extract_orb(batch):
    kpts, dess = [], []
    for i in batch:
        k, d = orb.detectAndCompute(i, None)
        kpts.append(k)
        dess.append(d)
    return kpts, dess

_superpoint = SuperPointFrontend('./SuperPointPretrainedNetwork/superpoint_v1.pth', 4, 0.015, 0.7, True)
def extract_superpoint(batch):
    pts, desc, heatmap = _superpoint.run(batch)
    return pts, desc

