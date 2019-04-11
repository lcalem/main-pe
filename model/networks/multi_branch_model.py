import time

import tensorflow as tf

from tensorflow.keras import Model, Input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop

from model.losses import pose_loss, vgg_loss, reconstruction_loss

from model.networks import BaseModel
from model.networks.decoder_model import DecoderModel
from model.networks.pose_model import PoseModel


class MultiBranchModel(BaseModel):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    One common decoder to recreate the image
    '''

    def __init__(self, dim, n_joints=16, nb_pose_blocks=8, reception_kernel_size=(5, 5), verbose=True):
        assert dim in [2, 3], 'Cannot work outside of 2D or 3D'
        
        self.dim = dim
        self.n_joints = n_joints
        self.n_blocks = nb_pose_blocks
        self.reception_kernel_size = reception_kernel_size
        self.verbose = verbose

        BaseModel.__init__(self)
        
    def load_weights(self, weights_path, pose_only=False):
        if pose_only:
            self.build_pose_only()
        else:
            self.build()
        self.model.load_weights(weights_path)

    def log(self, msg):
        if self.verbose:
            print(msg)
        
    def build(self):
        
        # build everything
        time_1 = time.time()
        self.appearance_model = self.build_appearance_model(self.input_shape)
        time_2 = time.time()
        self.pose_model = self.build_pose_model(self.input_shape)
        time_3 = time.time()
        self.decoder_model = self.build_decoder_model((16, 16, 2048))  # ...
        time_4 = time.time()
        
        self.log("Build E_a %s, build E_p %s, decoder D %s" % (time_2 - time_1, time_3 - time_2, time_4 - time_3))
        
        inp = Input(shape=self.input_shape)
        self.log("Input shape %s" % str(inp.shape))
        
        # encoders
        z_a = self.appearance_model(inp)
        assert z_a.shape.as_list() == [None, 16, 16, 1024], 'wrong shape for z_a %s' % str(z_a.shape.as_list())
        pose_outputs = self.pose_model(inp)

        poses, z_p = self.check_pose_output(pose_outputs)
        self.log("Shape z_a %s, shape z_p %s" % (str(z_a.shape), str(z_p.shape)))

        # decoder
        concat = self.concat(z_a, z_p)
        assert concat.shape.as_list() == [None, 16, 16, 2048], 'wrong concat shape %s' % str(concat.shape)
        i_hat = self.decoder_model(concat)

        # losses
        pose_losses = [pose_loss()] * self.n_blocks
        losses = [reconstruction_loss()] + pose_losses
        
        # model
        outputs = [i_hat]
        outputs.extend(poses)
        self.model = Model(inputs=inp, outputs=outputs)
        self.log("Outputs shape %s" % self.model.output_shape)
        
        self.model.compile(loss=losses, optimizer=RMSprop(lr=self.start_lr))
        
        if self.verbose:
            self.model.summary()
        
    def build_pose_only(self):
        '''
        Only the pose branch will be built and activated, no concat, no decoder
        -> for baselines and ablation study
        '''
        self.model = self.build_pose_model(self.input_shape, pose_only=True)
        
        ploss = [pose_loss()] * self.n_blocks
        self.model.compile(loss=ploss, optimizer=RMSprop(lr=self.start_lr))
        
        if self.verbose:
            self.model.summary()

    def build_appearance_model(self, input_shape):
        '''
        resnet50 for now
        input: 256 x 256 x 3
        output: 16 x 16 x 1024
        '''
        enc_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        output_layer = enc_model.layers[-33]  # index of the 16 x 16 x 1024 activation we want, before the last resnet block
        assert output_layer.name.startswith('activation')
        
        partial_model = Model(inputs=enc_model.inputs, outputs=output_layer.output)
        return partial_model
    
    def build_pose_model(self, input_shape, pose_only=False):
        '''
        reception / stacked hourglass
        input: 256 x 256 x 3
        output: [(n_joints, dim + 1) * n_blocks, (16, 16, 1024)]
        '''
        return PoseModel(input_shape, self.dim, self.n_joints, self.n_blocks, self.reception_kernel_size, pose_only=pose_only, verbose=self.verbose).model
    
    def check_pose_output(self, pose_outputs):
        '''
        pose_outputs should be a list of poses + z_p
        '''
        assert len(pose_outputs) == self.n_blocks + 1  # + 1 for the z_p (16 x 16 x 1024) representation
        poses = pose_outputs[:-1]
        z_p = pose_outputs[-1]
        
        pose_shapes = [pose.shape.as_list() for pose in poses]
        assert all([shape == [None, self.n_joints, self.dim + 1] for shape in pose_shapes]), 'pose shapes are weird %s' % str(pose_shapes)
        assert z_p.shape.as_list() == [None, 16, 16, 1024], 'z_p shape not as expected %s' % str(z_p.shape.as_list())
        
        return poses, z_p

    def concat(self, z_a, z_p):
        '''
        concat pose and appearance representations before decoding
        input:
            - z_p: 16 x 16 x 1024
            - z_a: 16 x 16 x 1024
        output:

        TODO: This is where the real work should happen
        '''
        concat = concatenate([z_a, z_p])
        return concat

    def build_decoder_model(self, input_shape):
        '''
        from concatenated representations to image reconstruction
        input: 16 x 16 x 2048 (z_a and z_p concatenated)
        output: 256 x 256 x 3
        '''
        return DecoderModel(input_shape=input_shape).model
    
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
