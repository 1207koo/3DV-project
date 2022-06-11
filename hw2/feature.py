import time
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

ransac_n_iter = 200
ransac_thr = 1e-2

def convert_to_homo(p):
    shape = p.shape
    t = np.ones((*shape[:-1], shape[-1] + 1))
    t[..., :-1] = p
    return t

def convert_to_inhomo(p):
    return p[..., :-1] / p[..., -1:]

def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    def nn(spt, qry):
        neigh = NearestNeighbors()
        neigh.fit(spt)
        dist, ind = neigh.kneighbors(qry, 2, return_distance=True)

        # ratio test
        return [(i, ind[i][0]) for i, (d1, d2) in enumerate(dist) if d1 < d2 * 0.7]

    ind_1to2 = nn(des2, des1)
    ind_2to1 = nn(des1, des2)

    # bi-directional consistency
    ind = set(ind_1to2) & set([(b, a) for a, b in ind_2to1])
    ind = np.array(list(ind))

    x1 = loc1[ind[:, 0]]
    x2 = loc2[ind[:, 1]]

    return x1, x2, ind[:, 0]



def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    # normalize
    """
    c1 = np.mean(x1, axis=0)
    c2 = np.mean(x2, axis=0)

    rms1 = np.sqrt(np.mean((x1 - c1[None, ...]) ** 2, axis=0))
    rms2 = np.sqrt(np.mean((x2 - c2[None, ...]) ** 2, axis=0))

    transM1 = np.zeros((3,3))
    transM1[:,-1] = [*c1, 1]
    transM2 = np.zeros((3,3))
    transM2[:, -1] = [*c2, 1]

    scaleM1 = np.diag([*(1. / rms1 * (2 ** 0.5)), 1])
    scaleM2 = np.diag([*(1. / rms2 * (2 ** 0.5)), 1])

    T1 = scaleM1 @ transM1
    T2 = scaleM2 @ transM2

    x1 = (T1 @ convert_to_homo(x1).T).T
    x2 = (T2 @ convert_to_homo(x2).T).T
    """

    N = len(x1)
    A = []
    for i in range(N):
        x1i = x1[i]
        x2i = x2[i]
        A.append([x2i[0] * x1i[0], x2i[0] * x1i[1], x2i[0], 
                  x2i[1] * x1i[0], x2i[1] * x1i[1], x2i[1], 
                  x1i[0], x1i[1], 1])
    A = np.asarray(A)

    u, s, vh = np.linalg.svd(A)
    E = vh[-1].reshape(3, 3)
    
    u, s, vh = np.linalg.svd(E)
    # clean up E
    #lmbda = (s[0, 0] + s[1, 1]) / 2.
    #s = np.diag([lmbda, lmbda, 0])
    s = np.diag([1, 1, 0])
    E = u @ s @ vh

    # unnormalize
    #E = T2.T @ E @ T1
    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    
    def count_inlier(E, p1, p2):
        p1 = convert_to_homo(p1)
        p2 = convert_to_homo(p2)

        diff = np.sum(p2 * (E @ p1.T).T, axis=1)
        ids = np.nonzero(np.abs(diff) < ransac_thr)[0]
        return len(ids), ids

    best_E = None
    best_inids = None
    best_count = 0
    for _ in range(ransac_n_iter):
        i = np.random.choice(len(x1), size=(8), replace=False)
        p1, p2 = x1[i], x2[i]
        E = EstimateE(p1, p2)
        count, ids = count_inlier(E, x1, x2)

        if count > best_count:
            best_E = E
            best_inids = ids
            best_count = count

    p1, p2 = x1[best_inids], x2[best_inids]
    E = EstimateE(p1, p2)
    return E, best_inids


elapsed = 0
nmatches = 0
n = 0

def BuildFeatureTrack(Im, K):
    global elapsed
    global nmatches
    global n
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    N = len(Im)
    inv_K = np.linalg.inv(K)

    # Extract SIFT descriptor
    #method = 'sift'
    method = 'd2net'
    method = 'ours'
    if method == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        des = []
        loc = []
        for i in range(N):
            k, d = sift.detectAndCompute(Im[i], None)
            des.append(d)
            loc.append(np.array([_k.pt for _k in k]))
    else:
        import torch
        if method == 'd2net':
            from d2_net.lib.model_test import D2Net
            model = D2Net(
                model_file='d2_net/models/d2_tf.pth',
                use_relu=True,
                use_cuda=False
            )
            from d2_net.lib.utils import preprocess_image
            from d2_net.lib.pyramid import process_multiscale

            loc = []
            des = []
            for i in range(N):
                with torch.no_grad():
                    input_image = preprocess_image(
                        Im[i],
                        preprocessing='caffe'
                    )
                    keypoints, scores, descriptors = process_multiscale(
                                    torch.tensor(
                                        input_image[np.newaxis, :, :, :].astype(np.float32),
                                        device='cpu'
                                    ),
                                    model,
                                    scales=[1]
                                )
                # i, j -> u, v
                keypoints = keypoints[:, [1, 0, 2]]
                loc.append(np.asarray([k[:2] for k in keypoints]))
                des.append(descriptors)      
        else:
            from model import D3Net
            model = D3Net()
            model.load_state_dict(torch.load('run070_pious-frost-70_epoch084.pt', map_location='cpu'))
            from d2_net.lib.utils import preprocess_image
            from util import process_multiscale_d3

            loc = []
            des = []
            for i in range(N):
                with torch.no_grad():
                    input_image = preprocess_image(
                        Im[i],
                        preprocessing='caffe'
                    )
                    keypoints, scores, descriptors = process_multiscale_d3(
                                    torch.tensor(
                                        input_image[np.newaxis, :, :, :].astype(np.float32),
                                        device='cpu'
                                    ),
                                    model,
                                    scales=[1]
                                )
                # i, j -> u, v
                keypoints = keypoints[:, [1, 0, 2]]
                loc.append(np.asarray([k[:2] for k in keypoints]))
                des.append(descriptors)
        
    # Build Track
    track = []
    for i in range(N):
        desi = des[i]
        loci = loc[i]
        track_tmp = -1 * np.ones((N, len(desi), 2))
        indis = set()
        for j in range(i + 1, N):
            desj = des[j]
            locj = loc[j]
            st = time.time()
            xi, xj, indi = MatchSIFT(loci, desi, locj, desj)
            elapsed += time.time() - st
            nmatches += len(indi)
            n += 1 # len(loci) * len(locj)

            xi = convert_to_inhomo((inv_K @ convert_to_homo(xi).T).T)
            xj = convert_to_inhomo((inv_K @ convert_to_homo(xj).T).T)
            
            E, inlier = EstimateE_RANSAC(xi, xj, ransac_n_iter, ransac_thr)
            
            track_tmp[i, indi[inlier]] = xi[inlier]
            track_tmp[j, indi[inlier]] = xj[inlier]
            indis.update(set(indi[inlier].tolist()))
        print(len(indis))
        
        track.append(track_tmp[:, list(indis)])
    return np.concatenate(track, axis=1)
