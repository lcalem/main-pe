import argparse
import datetime
import os
import sys

sys.path.append("%s/main-pe/" % os.environ['HOME'])   # re-lol

from data.datasets.h36m import Human36M
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE
from data.loader import BatchLoader

from experiments.common import exp_init

from model import config
from model.networks.multi_branch_model import MultiBranchModel
from model.utils import pose_format


def launch_training(dataset_path, model_folder, n_epochs, batch_size, pose_blocks):

    h36m_path = dataset_path
    h36m = Human36M(h36m_path, dataconf=config.human36m_dataconf, poselayout=pose_format.pa17j3d, topology='frames')

    data_tr_h36m = BatchLoader(
        h36m, 
        ['frame'], 
        ['frame'] + ['pose'] * pose_blocks,
        TRAIN_MODE, 
        batch_size=batch_size,
        shuffle=True)

    model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=pose_blocks)
    model.build()

    model.train(data_tr_h36m, steps_per_epoch=len(data_tr_h36m), model_folder=model_folder, n_epochs=n_epochs)


# python3 hybrid_h36m.py --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --model_name '' --n_epochs 60 --batch_size 16 --pose_blocks 1 --gpu 2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--pose_blocks", type=int, default=4)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    
    filename = os.path.basename(__file__).split('.')[0]
    model_folder = exp_init(filename, vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launch_training(args.dataset_path, model_folder, int(args.n_epochs), args.batch_size, args.pose_blocks)

