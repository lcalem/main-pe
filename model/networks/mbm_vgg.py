import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Lambda

from model.losses import pose_loss, vgg_loss

from model.networks.multi_branch_model import MBMBase


class MultiBranchVGGModel(MBMBase):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    One common decoder to recreate the image
    '''

    def __init__(self, dim, n_joints=16, nb_pose_blocks=8, reception_kernel_size=(5, 5), verbose=True):

        MBMBase.__init__(self, dim, n_joints, nb_pose_blocks, reception_kernel_size, verbose, zp_depth=1024, za_depth=1024)
        
    def get_losses_outputs(self, i_hat, poses):
        '''
        VGG loss (perceptual loss) for i_hat reconstruction
        '''
        # losses
        vgg_model = self.build_vgg_model(i_hat.shape.as_list()[1:])
        vgg_rec_outputs = vgg_model(i_hat)
        vgg_ori_outputs = vgg_model(inp)

        vgg_outputs = [Lambda(self.stack_vgg_outputs)([vgg_rec_outputs[i], vgg_ori_outputs[i]]) for i in range(len(vgg_rec_outputs))]
       
        print("Type vgg output %s" % type(vgg_outputs))
        for i, out in enumerate(vgg_outputs):
            print("Shape layer %s: %s" % (i, out.shape))
    
        pose_losses = [pose_loss()] * self.n_blocks
        vgg_losses = [vgg_loss()] * len(vgg_outputs)

        losses = vgg_losses + pose_losses
        
        # outputs
        outputs = vgg_outputs
        outputs.extend(poses)
        
        return losses, outputs
    
    def build_vgg_model(self, input_shape):
        '''
        VGG model for perceptual loss
        input: 256 x 256 x 3 reconstructed image (i_hat)
        output: 
        '''
        vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        vgg_model.trainable=False

        for layer in vgg_model.layers:
            layer.trainable=False
            
        # output_layers = [1,3,4,6,7]
        output_layers = [1, 4, 7]  # block1_conv1, block2_conv1, block3_conv1
        outputs = [vgg_model.layers[i].output for i in output_layers]
        
        vgg_loss_model = Model(vgg_model.inputs, outputs)
        return vgg_loss_model
    
    def stack_vgg_outputs(self, vgg_outputs):
        return tf.stack([vgg_outputs[0], vgg_outputs[1]], axis=1)
