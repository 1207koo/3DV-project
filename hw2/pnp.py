import numpy as np

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation

def convert_to_homo(p):
    shape = p.shape
    t = np.ones((*shape[:-1], shape[-1] + 1))
    t[..., :-1] = p
    return t

def convert_to_inhomo(p):
    return p[..., :-1] / p[..., -1:]

def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """

    vid1 = np.nonzero(1 - np.prod(X == -1, axis=-1))[0]
    vid2 = np.nonzero(1 - np.prod(x == -1, axis=-1))[0]
    valid_index = np.asarray(list(set(vid1.tolist()) & set(vid2.tolist())))
    X = X[valid_index]
    x = x[valid_index]

    A = []
    for i in range(len(X)):
        lx = X[i]
        sx = x[i]

        cnst1 = [lx[0], lx[1], lx[2], 1, 0, 0, 0, 0, -sx[0] * lx[0], -sx[0] * lx[1], -sx[0] * lx[2], -sx[0]]
        cnst2 = [0, 0, 0, 0, lx[0], lx[1], lx[2], 1, -sx[1] * lx[0], -sx[1] * lx[1], -sx[1] * lx[2], -sx[1]]
        A.append(cnst1)
        A.append(cnst2)

    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A)
    P = vh[-1].reshape(3, 4)

    #tmp = convert_to_inhomo((P @ convert_to_homo(X).T).T)

    lP = P[:, :-1]
    rP = P[:, -1]

    u, s, vh = np.linalg.svd(lP)
    R = u @ vh
    t = 1. / (s[0]) * rP
    #t = 1. / (s.mean()) * rP
    C = -R.T @ t

    if np.linalg.det(R) < 0:
        R = -R

    P2 = np.concatenate([R, -R @ C.reshape(3, 1)], axis=-1)
    #tmp2 = convert_to_inhomo((P2 @ convert_to_homo(X).T).T)
    #print(np.stack([np.linalg.norm(tmp - x, axis=-1), np.linalg.norm(tmp2 - x, axis=-1)], axis=-1))
    
    return R, C



def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    
    def count_inlier(R, C, X, x):
        P = np.concatenate([R, -R @ C.reshape(3, 1)], axis=-1)
        X = convert_to_homo(X)
        x_pred = (P @ X.T).T
        ids1 = x_pred[:, -1] > 0

        x_pred = convert_to_inhomo(x_pred)
        dist = np.linalg.norm(x - x_pred, axis=-1)
        ids2 = dist < ransac_thr
    
        ids = ids1 * ids2
        return np.sum(ids), ids

    N = len(X)

    vid1 = np.nonzero(1 - np.prod(X == -1, axis=-1))[0]
    vid2 = np.nonzero(1 - np.prod(x == -1, axis=-1))[0]
    valid_index = np.asarray(list(set(vid1.tolist()) & set(vid2.tolist())))
    print(valid_index.shape)
    X = X[valid_index]
    x = x[valid_index]

    best_RC = None
    best_count = 0
    best_inds = None
    for _ in range(ransac_n_iter):
        i = np.random.choice(len(X), size=(6), replace=False)
        lx = X[i]
        sx = x[i]
        R, C = PnP(lx, sx)
        count, ids = count_inlier(R, C, X, x)

        #print(count)
        if count > best_count:
            best_count = count
            best_RC = (R, C)
            best_inids = ids

    R, C = best_RC
    inlier = np.zeros(N, dtype=bool)
    inlier[valid_index[best_inids]] = True
    print("PnP Ransac: {}".format(best_count * 100. / len(X)))

    return R, C, inlier



def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0,:]
    dv_dc = -R[1,:]
    dw_dc = -R[2,:]
    # df_dc is in shape (2, 3)
    df_dc = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    # du_dR = np.concatenate([X-C, np.zeros(3), X-C])
    # dv_dR = np.concatenate([np.zeros(3), X-C, X-C])
    # dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    du_dR = np.concatenate([X-C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X-C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack([
        (w * du_dR - u * dw_dR) / (w**2),
        (w * dv_dR - v * dw_dR) / (w**2)
    ], axis=0)


    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray([
        [0, 0, -4*qy, -4*qz],
        [-2*qz, 2*qy, 2*qx, -2*qw],
        [2*qy, 2*qz, 2*qw, 2*qx],
        [2*qz, 2*qy, 2*qx, 2*qw],
        [0, -4*qx, 0, -4*qz],
        [-2*qx, -2*qw, 2*qz, 2*qy],
        [-2*qy, 2*qz, -2*qw, 2*qx],
        [2*qx, 2*qw, 2*qz, 2*qy],
        [0, -4*qx, -4*qy, 0],
    ])

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis,:]) @ R_i.T
        proj = proj[:,:2] / proj[:,2,np.newaxis]

        H = np.zeros((7,7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j,:])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j,:] - proj[j,:])
        
        delta_p = np.linalg.inv(H + lamb*np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)


    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined
