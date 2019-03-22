import argparse
import datetime
import os
import shutil
import yaml


def exp_init(exp_file, params):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - dump current model code in it (because for now we only save weights) TODO
    '''
    # model folder
    model_folder = '%s/pe_experiments/exp_%s_%s_%s_%sb_bs%s' % (os.environ['HOME'], datetime.datetime.now().strftime("%Y%m%d_%H%M"), exp_file, params['model_name'], params['pose_blocks'], params['batch_size'])
    os.makedirs(model_folder)
    print("Conducting experiment for %s epochs and %s blocks in folder %s" % (params['n_epochs'], params['pose_blocks'], model_folder))

    # config
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(params, f_conf, default_flow_style=False)
        
    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/../model'
    dst_folder = os.path.join(model_folder, 'model_src/')
    shutil.copytree(src_folder, dst_folder)
    
    return model_folder
    
        
        