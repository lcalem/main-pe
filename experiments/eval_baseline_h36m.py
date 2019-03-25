import numpy as np
import tensorflow as tf

from pprint import pprint

from model import config
from model.utils import pose_format

from data.datasets.h36m import Human36M
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE


def eval_human36m_sc_error(
    model, 
    x, 
    pose_w, 
    afmat, 
    rootz, 
    scam, 
    resol_z=2000., 
    batch_size=8, 
    map_to_pa17j=None, 
    logdir=None,
    verbose=True):

    assert len(x) == len(pose_w) == len(afmat) == len(scam)

    input_shape = model.input_shape
    num_blocks = len(model.outputs)

    y_true_w = pose_w.copy()
    if map_to_pa17j is not None:
        y_true_w = y_true_w[:, map_to_pa17j, :]
    y_pred_w = np.zeros((num_blocks,) + y_true_w.shape)
    if rootz.ndim == 1:
        rootz = np.expand_dims(rootz, axis=-1)

    pred = model.predict(x, batch_size=batch_size, verbose=1)

    # Move the root joints from GT poses to origin
    y_true_w -= y_true_w[:,0:1,:]

    if verbose:
        printc(WARNING, 'Avg. mm. error:')

    lower_err = np.inf
    lower_i = -1
    scores = []

    for b in range(num_blocks):

        if num_blocks > 1:
            y_pred = pred[b]
        else:
            y_pred = pred

        # ??
        y_pred = y_pred[:, :, 0:3]

        # Project normalized coordiates to the image plane
        y_pred[:, :, 0:2] = transform_pose_sequence(afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        """Recover the absolute Z."""
        y_pred[:, :, 2] = (resol_z * (y_pred[:, :, 2] - 0.5)) + rootz
        if map_to_pa17j is not None:
            y_pred_uvd = y_pred[:, map_to_pa17j, 0:3]
        else:
            y_pred_uvd = y_pred[:, :, 0:3]

        """Do the camera inverse projection."""
        for j in range(len(y_pred_uvd)):
            cam = camera_deserialize(scam[j])
            y_pred_w[b, j, :, :] = cam.inverse_project(y_pred_uvd[j])

        """Move the root joint from predicted poses to the origin."""
        y_pred_w[b, :, :, :] -= y_pred_w[b, :, 0:1, :]

        err_w = mean_distance_error(y_true_w[:, 0:, :], y_pred_w[b, :, 0:, :])
        scores.append(err_w)
        if verbose:
            printc(WARNING, ' %.1f' % err_w)

        """Keep the best prediction and its index."""
        if err_w < lower_err:
            lower_err = err_w
            lower_i = b

    if verbose:
        printcn('', '')

    if logdir is not None:
        np.save('%s/y_pred_w.npy' % logdir, y_pred_w)
        np.save('%s/y_true_w.npy' % logdir, y_true_w)

    printcn(WARNING, 'Final averaged error (mm): %.3f' % lower_err)

    return scores


def eval_baseline_h36m():
    weights_path = '/home/caleml/pe_experiments/exp_baseline_1b_bs32_h36m_201903221052/weights_032.h5'
    eval_model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=1)
    eval_model.load_weights(weights_path, pose_only=True)
    
    # local loading
    local_h36m_path = '/home/caleml/datasets/h36m'
    local_h36m = Human36M(local_h36m_path, dataconf=config.human36m_dataconf, poselayout=pose_format.pa17j3d, topology='frames')
    
    h36m_val = BatchLoader(local_h36m, 
                       ['frame'],
                       ['pose_w', 'pose_uvd', 'afmat', 'camera'], 
                       VALID_MODE,
                       batch_size=local_h36m.get_length(VALID_MODE), 
                       shuffle=True)

    printcn(OKBLUE, 'Preloading Human3.6M validation samples...')

    [x_val], [pw_val, puvd_val, afmat_val, scam_val] = h36m_val[0]
    
    scores = eval_human36m_sc_error(eval_model.model, 
                                x_val, 
                                pw_val, 
                                afmat_val,
                                puvd_val[:,0,2], 
                                scam_val,  
                                batch_size=24)

    pprint(scores)