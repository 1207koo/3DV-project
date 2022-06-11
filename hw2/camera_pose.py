import numpy as np

from feature import EstimateE_RANSAC

ransac_n_iter = 200
ransac_thr = 1e-2

def convert_to_homo(p):
    shape = p.shape
    t = np.ones((*shape[:-1], shape[-1] + 1))
    t[..., :-1] = p
    return t

def convert_to_inhomo(p):
    return p[..., :-1] / p[..., -1:]

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """

    u, s, vh = np.linalg.svd(E)
    t1 = u[:, -1]
    t2 = -t1

    w = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r1 = u @ w @ vh
    r2 = u @ w.T @ vh
    if np.linalg.det(r1) < 0:
        r1 = -r1
    if np.linalg.det(r2) < 0:
        r2 = -r2

    R_set = np.stack([r1, r1, r2, r2], axis=0)
    C_set = np.stack([-r1.T @ t1, -r1.T @ t2, -r2.T @ t1, -r2.T @ t2], axis=0)

    return R_set, C_set


def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    valid_ind1 = np.nonzero(1 - np.prod(track1 == -1, axis=-1))[0]
    valid_ind2 = np.nonzero(1 - np.prod(track2 == -1, axis=-1))[0]

    valid_ind = list(set(valid_ind1.tolist()) & set(valid_ind2.tolist()))

    x1 = track1[valid_ind]
    x2 = track2[valid_ind]

    X = -1 * np.ones((len(track1), 3))
    for i in range(len(x1)):
        l1 = x1[i, 0] * P1[-1, :] - P1[0, :]
        l2 = x1[i, 1] * P1[-1, :] - P1[1, :]
        l3 = x2[i, 0] * P2[-1, :] - P2[0, :]
        l4 = x2[i, 1] * P2[-1, :] - P2[1, :]
        A = np.asarray([l1, l2, l3, l4])
        u, s, vh = np.linalg.svd(A)
        X[valid_ind[i]] = convert_to_inhomo(vh[-1])
    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    X = convert_to_homo(X)
    X1 = (P1 @ X.T).T
    X2 = (P2 @ X.T).T

    vid1 = X1[:, -1] > 0
    vid2 = X2[:, -1] > 0

    valid_index = vid1 * vid2

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    valid_ind1 = np.nonzero(1 - np.prod(track1 == -1, axis=-1))[0]
    valid_ind2 = np.nonzero(1 - np.prod(track2 == -1, axis=-1))[0]

    valid_ind = list(set(valid_ind1.tolist()) & set(valid_ind2.tolist()))

    x1 = track1[valid_ind]
    x2 = track2[valid_ind]

    E, inlier = EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr)

    Rs, Cs = GetCameraPoseFromE(E)
    best_count = 0
    best_i = None
    best_X = None
    best_vid = None
    for i in range(4):
        P1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
        P2 = np.concatenate([Rs[i], -Rs[i] @ Cs[i].reshape(3, 1)], axis=-1)
        X = Triangulation(P1, P2, track1, track2) 
        vid = EvaluateCheirality(P1, P2, X)

        count = np.sum(vid)
        print("Camera estimate: {}, {}".format(i, count))
        if count > best_count:
            best_count = count
            best_i = i
            best_X = X
            best_vid = vid
    R = Rs[best_i]
    C = Cs[best_i]
    X = -1 * np.ones_like(best_X)
    X[best_vid] = best_X[best_vid]
    
    return R, C, X
