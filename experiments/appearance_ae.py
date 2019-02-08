import argparse
import datetime
import os
import sys

sys.path.append("/home/caleml/main-pe/")   # re-lol

from data.datasets.mpii import MpiiSinglePerson
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE
from data.loader import BatchLoader

from model import config
from model.models import AppearanceModel
from model.utils import pose_format


def launch_training(dataset_path, model_folder, n_epochs):

    mpii_path = dataset_path
    mpii = MpiiSinglePerson(mpii_path, dataconf=config.mpii_dataconf, poselayout=pose_format.pa17j3d)

    data_tr_mpii = BatchLoader(mpii, 
                               ['frame'], 
                               ['frame'], 
                               TRAIN_MODE,
                               shuffle=False)

    model = AppearanceModel()
    model.build()

    model.train(data_tr_mpii, steps_per_epoch=len(data_tr_mpii), model_folder=model_folder, n_epochs=n_epochs)


# python3 appearance_ae.py --dataset_path '/share/DEEPLEARNING/datasets/mpii' --dataset_name 'mpii' --model_name 'appearance' --n_epochs 60 --gpu 3
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_epochs", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()

    model_folder = '/home/caleml/pe_experiments/exp_%s_%s_%s' % (args.model_name, args.dataset_name, datetime.datetime.now().strftime("%Y%m%d%H%M")) 
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs in folder %s" % (args.n_epochs, model_folder))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    launch_training(args.dataset_path, model_folder, int(args.n_epochs))