import numpy as np

from data.utils import transform, camera
from model.utils import log


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


def eval_human36m_sc_error(model,
                           num_blocks,
                           x,
                           pose_w,
                           afmat,
                           rootz,
                           scam,
                           pose_only=False,
                           resol_z=2000.,
                           batch_size=8,
                           logdir=None,
                           verbose=True):

    assert len(x) == len(pose_w) == len(afmat) == len(scam)

    y_true_w = pose_w.copy()
    # if map_to_pa17j is not None:
    #     y_true_w = y_true_w[:, map_to_pa17j, :]
    
    y_pred_w = np.zeros((num_blocks,) + y_true_w.shape)
    if rootz.ndim == 1:
        rootz = np.expand_dims(rootz, axis=-1)

    pred = model.predict(x, batch_size=batch_size, verbose=1)

    # Move the root joints from GT poses to origin
    y_true_w -= y_true_w[:, 0:1, :]

    if verbose:
        log.printc(log.WARNING, 'Avg. mm. error:')

    lower_err = np.inf
    scores = []

    for b in range(num_blocks):

        if pose_only:
            y_pred = pred[b]
        else:
            y_pred = pred[b + 1]  # first output is image and pose output start after

        # ??
        y_pred = y_pred[:, :, 0:3]

        # project normalized coordiates to the image plane
        y_pred[:, :, 0:2] = transform.transform_pose_sequence(afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        # Recover the absolute Z
        y_pred[:, :, 2] = (resol_z * (y_pred[:, :, 2] - 0.5)) + rootz
        y_pred_uvd = y_pred[:, :, 0:3]

        # camera inverse projection
        for j in range(len(y_pred_uvd)):
            cam = camera.camera_deserialize(scam[j])
            y_pred_w[b, j, :, :] = cam.inverse_project(y_pred_uvd[j])

        # Move the root joint from predicted poses to the origin
        y_pred_w[b, :, :, :] -= y_pred_w[b, :, 0:1, :]

        err_w = mean_distance_error(y_true_w[:, 0:, :], y_pred_w[b, :, 0:, :])
        scores.append(err_w)
        if verbose:
            log.printc(log.WARNING, ' %.1f' % err_w)

        # Keep the best prediction
        if err_w < lower_err:
            lower_err = err_w

    if verbose:
        log.printcn('', '')

    if logdir is not None:
        np.save('%s/y_pred_w.npy' % logdir, y_pred_w)
        np.save('%s/y_true_w.npy' % logdir, y_true_w)

    log.printcn(log.WARNING, 'Final averaged error (mm): %.3f' % lower_err)

    return scores