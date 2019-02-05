import os

from tensorflow.keras.callbacks import Callback


class SaveModel(Callback):

    def __init__(self, filepath, verbose=True):

        self.filepath = filepath
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):

        filename = self.filepath.format(epoch=epoch + 1)
        
        if self.verbose:
            print('Saving model @epoch=%05d to %s' % (epoch + 1, filename))

        model.save_weights(filename)
