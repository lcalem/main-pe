import numpy as np


def _norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2), axis=axis))


def _valid_joints(y, min_valid=-1e6):
    def and_all(x):
        if x.all():
            return 1
        return 0

    return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))


def mean_distance_error(y_true, y_pred):
    '''
    Compute the mean absolute distance error on predicted samples, considering
    only the valid joints from y_true.

    y_true: [num_samples, nb_joints, dim]
    y_pred: [num_samples, nb_joints, dim]
    '''

    assert y_true.shape == y_pred.shape
    num_samples = len(y_true)

    dist = np.zeros(y_true.shape[0:2])
    valid = np.zeros(y_true.shape[0:2])

    for i in range(num_samples):
        valid[i, :] = _valid_joints(y_true[i])
        dist[i, :] = _norm(y_true[i] - y_pred[i], axis=1)

    match = dist * valid
    # print ('Maximum valid distance: {}'.format(match.max()))
    # print ('Average valid distance: {}'.format(match.mean()))

    return match.sum() / valid.sum()
