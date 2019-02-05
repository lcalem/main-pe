import numpy as np


def objposwin_to_bbox(objpos, winsize):
    x1 = objpos[0] - winsize[0]/2
    y1 = objpos[1] - winsize[1]/2
    x2 = objpos[0] + winsize[0]/2
    y2 = objpos[1] + winsize[1]/2

    return np.array([x1, y1, x2, y2])