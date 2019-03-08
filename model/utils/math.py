import numpy as np


def linspace_2d(nb_rols, nb_cols, dim=0):

    def _lin_sp_aux(size, nb_repeat, start, end):
        linsp = np.linspace(start, end, num=size)
        x = np.empty((nb_repeat, size), dtype=np.float32)

        for d in range(nb_repeat):
            x[d] = linsp

        return x

    if dim == 1:
        return (_lin_sp_aux(nb_rols, nb_cols, 0.0, 1.0)).T
    
    return _lin_sp_aux(nb_cols, nb_rols, 0.0, 1.0)