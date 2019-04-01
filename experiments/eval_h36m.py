import datetime
import os
import sys
import time
import yaml

from pprint import pprint

import numpy as np


def eval_human36m_sc_error(model,
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

    num_blocks = len(model.outputs)

    y_true_w = pose_w.copy()
    if map_to_pa17j is not None:
        y_true_w = y_true_w[:, map_to_pa17j, :]
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

        if num_blocks > 1:
            y_pred = pred[b]
        else:
            y_pred = pred

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

        err_w = measures.mean_distance_error(y_true_w[:, 0:, :], y_pred_w[b, :, 0:, :])
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


# whole folder eval
def eval_folder(exp_folder, dataset, pose_only=False):

    config_path = os.path.join(exp_folder, 'config.yaml')
    
    # config
    with open(config_path, 'r') as f_conf:
        config = yaml.load(f_conf)
        
    # model
    # eval_model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=int(config['pose_blocks']), verbose=False)
    eval_model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=int(config['pose_blocks']))
        
    # find weights paths
    weights = list()
    for filename in os.listdir(exp_folder):
        if filename.endswith(".h5") and filename.startswith("weights_"):
            weights.append(os.path.join(exp_folder, filename))
            
    weights.sort()
    print("Found %s weights paths" % len(weights))
    
    # actual eval
    all_scores = list()
    for i, weights_path in enumerate(weights[40:]):
        print("Eval of weights_path %s" % weights_path)
        eval_model.load_weights(weights_path, pose_only=pose_only)
        
        scores = eval_human36m_sc_error(eval_model.model, 
                                        dataset['x_val'], 
                                        dataset['pw_val'], 
                                        dataset['afmat_val'],
                                        dataset['puvd_val'][:,0,2], 
                                        dataset['scam_val'],  
                                        batch_size=24)
        
        print("Scores for epoch %s: %s" % (i + 41, str(scores)))
        all_scores.append(max(scores))
        
    return all_scores


def eval_h36m(exp_folder, pose_only):
    # local loading
    local_h36m_path = '/home/caleml/datasets/h36m'
    local_h36m = Human36M(local_h36m_path, dataconf=config.human36m_dataconf, poselayout=pose_format.pa17j3d, topology='frames')
    
    h36m_val = BatchLoader(local_h36m, 
                       ['frame'],
                       ['pose_w', 'pose_uvd', 'afmat', 'camera'], 
                       VALID_MODE,
                       batch_size=local_h36m.get_length(VALID_MODE), 
                       shuffle=True)

    log.printcn(log.OKBLUE, 'Preloading Human3.6M validation samples...')

    [x_val], [pw_val, puvd_val, afmat_val, scam_val] = h36m_val[0]
    
    dataset = {
        'x_val': x_val,
        'pw_val': pw_val,
        'puvd_val': puvd_val,
        'afmat_val': afmat_val,
        'scam_val': scam_val
    }
    
    all_scores_baseline = eval_folder(exp_folder, dataset, pose_only=pose_only)
    print(all_scores_baseline)
    
    
# python3 eval_h36m.py /home/caleml/pe_experiments/exp_baseline_1b_bs32_h36m_201903221052/  true
# python3 eval_h36m.py /home/caleml/pe_experiments/exp_20190322_1942_hybrid_h36m__1b_bs16/ false
if __name__ == '__main__':
    
    exp_folder = sys.argv[1]
    pose_only = sys.argv[2] == 'true'
    
    sys.path.append(os.path.join(exp_folder, 'model_src/'))

    from data.datasets.h36m import Human36M
    from data.loader import BatchLoader
    from data.utils import transform, camera
    from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE

    from model import config, measures
    from model.networks.multi_branch_model import MultiBranchModel
    from model.utils import pose_format, log
    
    eval_h36m(exp_folder, pose_only)
    