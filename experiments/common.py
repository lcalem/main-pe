import argparse
import datetime
import os
import shutil
import sys
import yaml

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard

sys.path.append("%s/main-pe/" % os.environ['HOME'])   # lol

from data.datasets.h36m import Human36M
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE
from data.loader import BatchLoader

from model import config
from model import callbacks
from model.networks.cycle_model import CycleModel
from model.networks.multi_branch_model import MultiBranchModel
from model.networks.mbm_vgg import MultiBranchVGGModel
from model.networks.mbm_reduced import MultiBranchReduced
from model.utils import pose_format, log


def exp_init(params):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - dump current model code in it (because for now we only save weights)
    '''
    # model folder
    model_folder = '%s/pe_experiments/exp_%s_%s_%s_%sb_bs%s' % (os.environ['HOME'], datetime.datetime.now().strftime("%Y%m%d_%H%M"), params['exp_type'], params.get('name', ''), params['pose_blocks'], params['batch_size'])
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs and %s blocks in folder %s" % (params['n_epochs'], params['pose_blocks'], model_folder))

    # config
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(params, f_conf, default_flow_style=False)
        
    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/..'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)
    
    return model_folder


def lr_scheduler(epoch, lr):

    if epoch in [20, 30]:
        newlr = 0.5 * lr
        log.printcn(log.WARNING, 'lr_scheduler: lr %g -> %g @ %d' % (lr, newlr, epoch))
    else:
        newlr = lr
        log.printcn(log.OKBLUE, 'lr_scheduler: lr %g @ %d' % (newlr, epoch))

    return newlr


class Launcher():
    
    def __init__(self, exp_type, dataset_path, model_folder, n_epochs, batch_size, pose_blocks, zp_depth):
        
        self.exp_type = exp_type
        self.dataset_path = dataset_path
        self.model_folder = model_folder
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pose_blocks = pose_blocks
        self.zp_depth = zp_depth
        
        if zp_depth is not None:
            assert exp_type == 'hybrid_reduced', 'zp_depth is an option for hybrid_reduced model'
        
        self.pose_only = True if exp_type == 'baseline' else False
        
        
    def launch(self):
        '''
        main entrypoint
        1. training data
        2. validation data
        3. callbacks
        4. model building
        5. launch actual training
        '''
        
        h36m = Human36M(self.dataset_path, dataconf=config.human36m_dataconf, poselayout=pose_format.pa17j3d, topology='frames')

        # training data
        dataset_output = self.get_h36m_outputs()
        data_tr_h36m = BatchLoader(
            h36m, 
            ['frame'], 
            dataset_output,
            TRAIN_MODE, 
            batch_size=self.batch_size,
            shuffle=True)
        
        # validation data
        h36m_val = BatchLoader(h36m, 
                               ['frame'], 
                               ['pose_w', 'pose_uvd', 'afmat', 'camera'], 
                               VALID_MODE, 
                               batch_size=h36m.get_length(VALID_MODE), 
                               shuffle=True)
        
        log.printcn(log.OKBLUE, 'Preloading Human3.6M validation samples...')
        [x_val], [pw_val, puvd_val, afmat_val, scam_val] = h36m_val[0]
    
        # callbacks
        cb_list = list()
        eval_callback = callbacks.H36MEvalCallback(self.pose_blocks, x_val, pw_val, afmat_val, puvd_val[:,0,2], scam_val, pose_only=self.pose_only, logdir=self.model_folder)
        
        logs_folder = os.environ['HOME'] + '/pe_experiments/tensorboard/' + self.model_folder.split('/')[-1]
        print('Tensorboard log folder %s' % logs_folder)
        tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        
        cb_list.append(tensorboard)
        cb_list.append(eval_callback)
        cb_list.append(LearningRateScheduler(lr_scheduler))
        cb_list.append(callbacks.SaveModel(self.model_folder))
        
        # model
        self.build_model()
        
        # train
        self.model.train(data_tr_h36m, steps_per_epoch=len(data_tr_h36m), model_folder=self.model_folder, n_epochs=self.n_epochs, cb_list=cb_list)
        
        
    def get_h36m_outputs(self):
        if self.exp_type == 'baseline':
            return ['pose'] * self.pose_blocks
        elif self.exp_type in ['hybrid', 'hybrid_reduced']:
            return ['frame'] + ['pose'] * self.pose_blocks
        elif self.exp_type == 'hybrid_vgg':
            return ['phony'] * 3 + ['pose'] * self.pose_blocks
        elif self.exp_type == 'cycle':
            return ['frame'] + ['pose'] * self.pose_blocks + ['phony'] * 3
        else:
            raise Exception('Unknown exp_type %s' % self.exp_type)
        
        
    def build_model(self):
        
        if self.exp_type == 'baseline':
            self.model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=self.pose_blocks)
            self.model.build_pose_only()
            
        elif self.exp_type == 'hybrid':
            self.model = MultiBranchModel(dim=3, n_joints=17, nb_pose_blocks=self.pose_blocks)
            self.model.build()
            
        elif self.exp_type == 'hybrid_reduced':
            assert isinstance(self.zp_depth, int), 'wrong zp_depth %s' % self.zp_depth
            log.printcn(log.OKBLUE, 'launching hybrid_reduced model with zp_depth = %s' % self.zp_depth)
            self.model = MultiBranchReduced(dim=3, n_joints=17, nb_pose_blocks=self.pose_blocks, zp_depth=self.zp_depth)
            self.model.build()
            
        elif self.exp_type == 'hybrid_vgg':
            self.model = MultiBranchVGGModel(dim=3, n_joints=17, nb_pose_blocks=self.pose_blocks)
            self.model.build()
            
        elif self.exp_type == 'cycle':
            self.model = CycleModel(dim=3, n_joints=17, nb_pose_blocks=self.pose_blocks)
            self.model.build()
    

# python3 common.py --exp_type hybrid --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 1 --gpu 2
# python3 common.py --exp_type hybrid --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 2 --gpu 2
# python3 common.py --exp_type cycle --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 12 --pose_blocks 2 --gpu 2

# python3 common.py --exp_type baseline --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 32 --pose_blocks 2 --gpu 1
# python3 common.py --exp_type hybrid --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 2 --gpu 2
# python3 common.py --exp_type hybrid_vgg --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 2 --gpu 0

# python3 common.py --exp_type hybrid_reduced --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 2 --gpu 2
# python3 common.py --exp_type hybrid_reduced --zp_depth 256 --dataset_path '/home/caleml/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 2 --gpu 3

## GPUSERVER3
# python3 common.py --exp_type baseline --dataset_path '/home/calem/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 3 --gpu 2
# python3 common.py --exp_type hybrid_reduced --dataset_path '/home/calem/datasets/h36m' --dataset_name 'h36m' --n_epochs 60 --batch_size 16 --pose_blocks 3 --gpu 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--pose_blocks", type=int)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--name")
    parser.add_argument("--zp_depth", type=int)
    args = parser.parse_args()
    
    model_folder = exp_init(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    launcher = Launcher(args.exp_type, args.dataset_path, model_folder, int(args.n_epochs), args.batch_size, args.pose_blocks, args.zp_depth)
    launcher.launch()
