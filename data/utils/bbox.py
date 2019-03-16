import numpy as np


def objposwin_to_bbox(objpos, winsize):
    x1 = objpos[0] - winsize[0]/2
    y1 = objpos[1] - winsize[1]/2
    x2 = objpos[0] + winsize[0]/2
    y2 = objpos[1] + winsize[1]/2

    return np.array([x1, y1, x2, y2])


def bbox_to_objposwin(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    wx = bbox[2] - bbox[0]
    wy = bbox[3] - bbox[1]

    return np.array([cx, cy]), (wx, wy)


def get_crop_params(rootj, imgsize, f, scale):
    assert len(rootj.shape) == 2 and rootj.shape[-1] == 3, 'Invalid rootj ' + 'shape ({}), expected (n, 3) vector'.format(rootj.shape)

    if len(rootj) == 1:
        idx = [0]
    else:
        idx = [0, int(len(rootj)/2 + 0.5), len(rootj)-1]

    x1 = y1 = np.inf
    x2 = y2 = -np.inf
    zrange = np.array([np.inf, -np.inf])
    for i in idx:
        objpos = np.array([rootj[0, 0], rootj[0, 1] + scale])
        d = rootj[0, 2]
        winsize = (2.25*scale)*max(imgsize[0]*f[0, 0]/d, imgsize[1]*f[0, 1]/d)
        bo = objposwin_to_bbox(objpos, (winsize, winsize))
        x1 = min(x1, bo[0])
        y1 = min(y1, bo[1])
        x2 = max(x2, bo[2])
        y2 = max(y2, bo[3])
        zrange[0] = min(zrange[0], d - scale*1000.)
        zrange[1] = max(zrange[1], d + scale*1000.)

    objpos, winsize = bbox_to_objposwin([x1, y1, x2, y2])

    return objpos, winsize, zrange
