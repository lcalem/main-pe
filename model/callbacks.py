import json
import os

from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import Callback

from model import measures
from model.utils import log


class SaveModel(Callback):

    def __init__(self, model_folder, verbose=True):

        self.model_folder = model_folder
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
    
        save_file = os.path.join(self.model_folder, 'model_{epoch:03d}.h5').format(epoch=epoch + 1)        
        try:
            if self.verbose:
                print('\nTrying to save model @epoch=%03d to %s' % (epoch + 1, save_file))
            
            save_model(self.model, save_file)
        except Exception as e:
            save_file = os.path.join(self.model_folder, 'weights_{epoch:03d}.h5').format(epoch=epoch + 1)
            print("Couldn't save model, saving weights instead at %s" % save_file)
            self.model.save_weights(save_file)

            
class H36MEvalCallback(Callback):

    def __init__(self, 
                 num_blocks,
                 x, 
                 pw, 
                 afmat, 
                 rootz, 
                 scam, 
                 pose_only=False,
                 batch_size=24, 
                 logdir=None):

        self.num_blocks = num_blocks
        self.x = x
        self.pw = pw
        self.afmat = afmat
        self.rootz = rootz
        self.scam = scam
        self.batch_size = batch_size
        self.pose_only = pose_only
        
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        model = self.model

        scores = measures.eval_human36m_sc_error(model, 
                                                 self.num_blocks,
                                                 self.x, 
                                                 self.pw, 
                                                 self.afmat,
                                                 self.rootz, 
                                                 self.scam, 
                                                 pose_only=self.pose_only,
                                                 batch_size=self.batch_size)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'h36m_val.json'), 'w') as f:
                f.write(json.dumps(self.logarray))

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        log.printcn(log.OKBLUE, 'Best score is %.1f at epoch %d' % (self.best_score, self.best_epoch))


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf