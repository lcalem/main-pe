import os

from tensorflow.keras.models import load_model

from model import callbacks
from model.utils import log


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.start_lr = 0.001
        
    def log(self, msg):
        if self.verbose:
            log.printcn(log.HEADER, msg)

    def load(self, checkpoint_path, custom_objects=None):
        self.model = load_model(checkpoint_path, custom_objects=custom_objects)
        
    def load_weights(self, weights_path, pose_only=False, by_name=False):
        if pose_only:
            self.build_pose_only()
        else:
            self.build()
        self.model.load_weights(weights_path, by_name=by_name)
        
    def build(self):
        raise NotImplementedError

    def train(self, data_tr, steps_per_epoch, model_folder, n_epochs, cb_list):
        
        print("Training with %s callbacks" % len(cb_list))

        self.model.fit_generator(data_tr,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=n_epochs,
                                 callbacks=cb_list,
                                 use_multiprocessing=False,
                                 max_queue_size=10,
                                 workers=2,
                                 initial_epoch=0)
        
    def predict(self, data):
        return self.model.predict(data)

    def evaluation(self, data_eval):
        pass
