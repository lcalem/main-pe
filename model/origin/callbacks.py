import os

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import Callback


class SaveModel(Callback):

    def __init__(self, model_folder, verbose=True):

        self.model_folder = model_folder
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):

        save_file = os.path.join(self.model_folder, 'model_{epoch:03d}.h5').format(epoch=epoch + 1)
        
        if self.verbose:
            print('Saving model @epoch=%03d to %s' % (epoch + 1, save_file))

        try:
            save_model(self.model, save_file)
        except Exception as e:
            save_file = os.path.join(self.model_folder, 'weights_{epoch:03d}.h5').format(epoch=epoch + 1)
            print("Couldn't save model, saving weights instead at %s" % save_file)
            self.model.save_weights(save_file)
