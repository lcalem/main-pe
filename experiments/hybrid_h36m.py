import argparse
import datetime
import os
import sys

sys.path.append("/home/caleml/main-pe/")   # re-lol

from data.datasets.mpii import MpiiSinglePerson
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE
from data.loader import BatchLoader

from model import config
from model.networks.multi_branch_model import MultiBranchModel


def launch_training(dataset_path, model_folder, n_epochs, batch_size):

    h36m_path = dataset_path
    h36m = Human36M(h36m_path, dataconf=config.human36m_dataconf, poselayout=pose_format.pa17j3d, topology='frames')

    data_tr_h36m = BatchLoader(
        h36m, 
        ['frame'], 
        ['frame', 'pose', 'pose', 'pose', 'pose'],
        TRAIN_MODE, 
        batch_size=16,
        shuffle=True)

    model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=4)
    model.build()

    model.train(data_tr_h36m, steps_per_epoch=len(data_tr_h36m), model_folder=model_folder, n_epochs=n_epochs)


# python3 separate_h36m.py --dataset_path '/share/DEEPLEARNING/datasets/h36m' --dataset_name 'h36m' --model_name 'separate_baseline' --n_epochs 60 --batch_size 16 --gpu 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_epochs", required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()

    model_folder = '/home/caleml/pe_experiments/exp_%s_%s_%s' % (args.model_name, args.dataset_name, datetime.datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs in folder %s" % (args.n_epochs, model_folder))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launch_training(dataset_path, model_folder, int(args.n_epochs), args.batch_size)

