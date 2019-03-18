import argparse
import sys

sys.path.append("/home/caleml/main-pe/")  # FIXME when docker

from experiments.common import exp_init


# python3 test.py --dataset_path '/share/DEEPLEARNING/datasets/h36m' --dataset_name 'TEST' --model_name 'TEST' --n_epochs 60 --batch_size 32 --pose_blocks 8 --gpu 0
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
    
    exp_init(args)
    
    