import tensorflow as tf

from tensorflow.keras import Model

from model.losses import pose_loss, reconstruction_loss

from model.networks.multi_branch_model import MBMBase
from model.networks.decoder_reduced import DecoderModel
from model.networks.contrib_resnet import ResNet18, ResNet34


class MultiBranchReduced(MBMBase):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    One common decoder to recreate the image
    '''

    def __init__(self, 
                 dim, 
                 n_joints=16, 
                 nb_pose_blocks=8, 
                 reception_kernel_size=(5, 5), 
                 verbose=True, 
                 zp_depth=128):
        '''
        zp_depth = the number of channels for the (16 x 16 x ?)  z_p representation outputted by the PoseModel
        In the regular form it is (16 x 16 x 1024) but here in the reduced form it is lower
        It will be concatenated with the reduced z_a of shape (16 x 16 x 128) here (instead of 1024 in the regular model)
        '''

        MBMBase.__init__(self, dim, n_joints, nb_pose_blocks, reception_kernel_size, verbose, zp_depth, za_depth=128)
        
    def get_losses_outputs(self, i_hat, poses):
        # losses
        pose_losses = [pose_loss()] * self.n_blocks
        losses = [reconstruction_loss()] + pose_losses
        
        # model
        outputs = [i_hat]
        outputs.extend(poses)
        
        return losses, outputs
        
    def build_appearance_model(self, input_shape):
        '''
        resnet18 for the reduced form
        input: 256 x 256 x 3
        output: 16 x 16 x 128
        '''
        enc_model = ResNet18(input_shape=input_shape, top=None)
        self.log("full ResNet18 model summary")
        enc_model.summary()
        output_layer = enc_model.layers[-33]  # index of the 16 x 16 x 128 activation we want, before the last resnet block (68 for ResNet34)
        assert output_layer.name.startswith('activation')
        
        partial_model = Model(inputs=enc_model.inputs, outputs=output_layer.output)
        self.log("partial appearance summary")
        partial_model.summary()
        return partial_model

    def build_decoder_model(self, input_shape):
        '''
        Here we use the reduced decoder model
        
        from concatenated representations to image reconstruction
        input: 16 x 16 x 256 (z_a and z_p concatenated)
        output: 256 x 256 x 3
        '''
        return DecoderModel(input_shape=input_shape).model
    