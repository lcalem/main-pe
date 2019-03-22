import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

from model import callbacks


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.start_lr = 0.001

    def load(self, checkpoint_path, custom_objects=None):
        self.model = load_model(checkpoint_path, custom_objects=custom_objects)
        
    def build(self):
        raise NotImplementedError

    def train(self, data_tr, steps_per_epoch, model_folder, n_epochs):

        cb_list = []
        cb_list.append(callbacks.SaveModel(model_folder))
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        # callbacks.append(eval_callback)
        
        # tensorboard
        logs_folder = os.environ['HOME'] + '/pe_experiments/tensorboard/' + model_folder.split('/')[-1]
        print('Tensorboard log folder %s' % logs_folder)
        tensorboard = TensorBoard(log_dir=os.path.join(logs_folder, 'tensorboard'))
        cb_list.append(tensorboard)

        self.model.fit_generator(data_tr,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=n_epochs,
                                 callbacks=cb_list,
                                 workers=4,
                                 initial_epoch=0)
        
    def predict(self, data):
        return self.model.predict(data)

    def evaluation(self, data_eval):
        pass
