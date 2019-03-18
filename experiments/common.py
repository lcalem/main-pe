import argparse
import datetime
import os
import shutil
import yaml


def exp_init(args):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - dump current model code in it (because for now we only save weights) TODO
    '''
    # model folder
    model_folder = '/home/caleml/pe_experiments/exp_%s_%s_%s' % (args.model_name, args.dataset_name, datetime.datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs and %s blocks in folder %s" % (args.n_epochs, args.pose_blocks, model_folder))

    # config
    config = vars(args)
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(config, f_conf, default_flow_style=False)
        
    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/../model'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)
    
        
        