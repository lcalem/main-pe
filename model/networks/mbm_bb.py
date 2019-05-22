import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import concatenate, Lambda
from tensorflow.keras.optimizers import RMSprop

from model.losses import pose_loss, reconstruction_loss

from model.networks.mbm_reduced import MultiBranchReduced


class MBMReducedBB(MultiBranchReduced):
    '''
    -> same as MBM reduced but we modify the build function to include the branch balancing thing

    '''

    def build(self):
        '''
        here only the line 'i_hat =' is modified
        '''
        
        # build everything
        self.build_everything()
        
        inp = Input(shape=self.input_shape, name='image_input')
        self.log("Input shape %s" % str(inp.shape))
        
        # encoders
        z_a, z_p, poses = self.call_encoders(inp)

        # decoder
        i_hat = Lambda(self.branch_balancing, name='i_hat_bb')([inp, z_a, z_p])

        # losses and outputs
        losses, outputs = self.get_losses_outputs(i_hat, poses)
        
        # model
        self.model = Model(inputs=inp, outputs=outputs)
        self.log("Outputs shape %s" % self.model.output_shape)
        
        self.model.compile(loss=losses, optimizer=RMSprop(lr=self.start_lr))
        
        if self.verbose:
            self.log("Final model summary")
            self.model.summary()
    
    def branch_balancing(self, bundle):
        '''
        1. concat z_a and zeros -> decoder -> i_hat_a
        2. concat z_p and zeros -> decoder -> i_hat_p
        3. compute loss L(i_hat_a, i) and L(i_hat_p, i)
        4. output only the i_hat where the loss is greater
        '''        
        inp, z_a, z_p = bundle
        
        assert z_a.get_shape().as_list() == z_p.get_shape().as_list()
        zeros = tf.zeros_like(z_a)
        
        concat_a = self.concat(z_a, zeros)
        i_hat_a = self.decoder_model(concat_a)
        print('shape i_hat_a %s' % str(i_hat_a.shape))
        
        concat_p = self.concat(zeros, z_p)
        i_hat_p = self.decoder_model(concat_p)
        print('shape i_hat_p %s' % str(i_hat_p.shape))
        
        # losses
        rec_loss_a = reconstruction_loss()(inp, i_hat_a)
        rec_loss_p = reconstruction_loss()(inp, i_hat_p)
        print('shape loss_a %s' % str(rec_loss_a.shape))
        print('shape loss_p %s' % str(rec_loss_p.shape))
        
        # mix of reconstructions using a or p only depending on which has the higher loss
        i_hat = tf.where(rec_loss_a > rec_loss_p, i_hat_a, i_hat_p)
        print('shape final i_hat %s' % str(i_hat.shape))
        return i_hat