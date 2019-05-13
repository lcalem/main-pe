import random
import time

import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import RMSprop

from model.losses import cycle_loss, noop_loss, pose_loss, reconstruction_loss
from model.networks.mbm_reduced import MultiBranchReduced


class CycleReduced(MultiBranchReduced):
    '''
    stopped gradients between decoder and Ep
    '''
    
    def build(self):
        '''
        override of build!
        -> first part (up to i_hat) is the same
        -> after i_hat we put the specific cycle thing
        
        TODO: refactor (uglyyyyyyy)
        
        Outputs for reduced cycle (in that order):
        - i_hat (None, 256, 256, 3)
        - pose (None, 17, 4)   (times nblock)
        - concat_z_a (None, 16, 16, 256)
        - concat_z_p (None, 16, 16, 256)
        - i_hat_shuffled (None, 256, 256, 3)
        '''
        
        # build everything
        self.build_everything()   # reduced decoder
        
        inp = Input(shape=self.input_shape, name='image_input')
        self.log("Input shape %s" % str(inp.shape))
        
        # encoders
        z_a, z_p, poses = self.call_encoders(inp)

        # decoder
        i_hat = self.concat_and_decode(z_a, z_p) 
        i_hat = Lambda(lambda x: x, name='i_hat')(i_hat)    # naming to differentiate from mixed

        # shuffle z_a and z_p from images from the batch and create new images
        concat_shuffled = self.shuffle(z_a, z_p)
        i_hat_mixed = self.decoder_model(concat_shuffled)
        i_hat_mixed = Lambda(lambda x: x, name='i_hat_mixed')(i_hat_mixed)
        
        # re-encode mixed images and get new z_a and z_p
        cycle_z_a = self.appearance_model(i_hat_mixed)
        cycle_pose_outputs = self.pose_model(i_hat_mixed)
        cycle_poses, cycle_z_p = self.check_pose_output(cycle_pose_outputs)
        
        # concat z_a and z_a', z_p and z_p' to have an output usable by the cycle loss
        concat_z_a = concatenate([z_a, cycle_z_a], name='cycle_za_concat')
        concat_z_p = concatenate([z_p, cycle_z_p], name='cycle_zp_concat')

        # build the whole model
        outputs = [i_hat] + poses + [concat_z_a] + [concat_z_p] + [i_hat_mixed]
        self.model = Model(inputs=inp, outputs=outputs)
        print("Outputs shape %s" % self.model.output_shape)

        ploss = [pose_loss()] * len(poses)
        losses = [reconstruction_loss()] + ploss + [cycle_loss(), cycle_loss(), noop_loss()]

        self.model.compile(loss=losses, optimizer=RMSprop(lr=self.start_lr))
        
        if self.verbose:
            self.log("Final model summary")
            self.model.summary()
            
    def shuffle(self, z_a, z_p):
        '''
        concat z_a and z_p (as self.concat does) but mixing z_a and z_p so they don't come from the same image
        input:
            - z_p: 16 x 16 x 128
            - z_a: 16 x 16 x 128
        output: 16 x 16 x 256
        '''
        # shuffled_z_p = Lambda(lambda x: tf.random.shuffle(x))(z_p)    NOT DIFF
            
        # indexes = list(range(self.batch_size)) 
        # random.shuffle(indexes)
        # mixed_z_p = z_p[indexes,:,:,:]
        # concat = concatenate([z_a, mixed_z_p])
           
        shuffled_z_p = Lambda(lambda x: tf.roll(x, shift=1, axis=0), name='shuffled_zp')(z_p)
        concat = concatenate([z_a, shuffled_z_p], name='concat_shuffle')
        sums_ori = tf.math.reduce_sum(z_p, axis=(1, 2, 3))
        sums_aft = tf.math.reduce_sum(shuffled_z_p, axis=(1, 2, 3))
        print("z_p sums %s" % str(sums_ori))
        print("mixed z_p sums %s" % str(sums_aft))
        return concat