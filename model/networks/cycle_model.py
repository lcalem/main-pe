import time

import tensorflow as tf

from tensorflow.keras import Model, Input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import RMSprop

from model.losses import cycle_loss, noop_loss, pose_loss, reconstruction_loss

from model.networks import BaseModel
from model.networks.decoder_model import DecoderModel
from model.networks.pose_model import PoseModel


class CycleModel(BaseModel):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    One common decoder to recreate the image
    
    Second decoding + encoding pass to disentangle z_a and z_p
    '''

    def __init__(self, dim, n_joints=16, nb_pose_blocks=8, reception_kernel_size=(5, 5)):
        assert dim in [2, 3], 'Cannot work outside of 2D or 3D'
        
        self.dim = dim
        self.n_joints = n_joints
        self.n_blocks = nb_pose_blocks
        self.reception_kernel_size = reception_kernel_size

        BaseModel.__init__(self)
        
    def load_weights(self, weights_path, pose_only=False):
        if pose_only:
            self.build_pose_only()
        else:
            self.build()
        self.model.load_weights(weights_path)

    def build(self):
        '''
        Outputs of this model is a ton of things so they can properly be used in losses. 
        Outputs in order:
        - first the reconstructed image i_hat (None, 256, 256, 3) -> reconstruction loss
        - then n_block * pose output (None, n_joints, dim + 1) (+1 for visibility prob) -> pose loss
        - then the concatenated z_a and z_a' -> cycle consistency loss
        - then the concatenated z_p and z_p' -> cycle consistency loss
        - then the intermediate mixed reconstructed images, for viz -> noop loss      
        '''
        
        # build everything
        time_1 = time.time()
        self.appearance_model = self.build_appearance_model(self.input_shape)
        time_2 = time.time()
        self.pose_model = self.build_pose_model(self.input_shape)
        time_3 = time.time()
        self.decoder_model = self.build_decoder_model((16, 16, 2048))  # ...
        time_4 = time.time()
        
        print("Build E_a %s, build E_p %s, decoder D %s" % (time_2 - time_1, time_3 - time_2, time_4 - time_3))
        
        inp = Input(shape=self.input_shape)
        print("Input shape %s" % str(inp.shape))

        # encoders
        z_a = self.appearance_model(inp)
        assert z_a.shape.as_list() == [None, 16, 16, 1024], 'wrong shape for z_a %s' % str(z_a.shape.as_list())
        pose_outputs = self.pose_model(inp)

        poses, z_p = self.check_pose_output(pose_outputs)
        print("Shape z_a %s, shape z_p %s" % (str(z_a.shape), str(z_p.shape)))

        # decoder
        concat = self.concat(z_a, z_p)
        assert concat.shape.as_list() == [None, 16, 16, 2048], 'wrong concat shape %s' % str(concat.shape)
        i_hat = self.decoder_model(concat)
        
        # shuffle z_a and z_p from images from the batch and create new images
        concat_shuffled = self.shuffle(z_a, z_p)
        i_hat_mixed = self.decoder_model(concat_shuffled)
        
        # re-encode mixed images and get new z_a and z_p
        cycle_z_a = self.appearance_model(i_hat_mixed)
        cycle_pose_outputs = self.pose_model(i_hat_mixed)
        cycle_poses, cycle_z_p = self.check_pose_output(cycle_pose_outputs)
        
        # concat z_a and z_a', z_p and z_p' to have an output usable by the cycle loss
        concat_z_a = concatenate([z_a, cycle_z_a])
        concat_z_p = concatenate([z_p, cycle_z_p])

        # build the whole model
        outputs = [i_hat] + poses + [concat_z_a] + [concat_z_p] + [i_hat_mixed]
        self.model = Model(inputs=inp, outputs=outputs)
        print("Outputs shape %s" % self.model.output_shape)

        ploss = [pose_loss()] * len(poses)
        losses = [reconstruction_loss()] + ploss + [cycle_loss(), cycle_loss(), noop_loss()]
        # loss = mean_squared_error
        self.model.compile(loss=losses, optimizer=RMSprop(lr=self.start_lr))
        self.model.summary()
        
    def build_pose_only(self):
        '''
        Only the pose branch will be built and activated, no concat, no decoder
        -> for baselines and ablation study
        '''
        inp = Input(shape=self.input_shape)
        z_p = self.pose_encoder(inp)
        
        self.model = Model(inputs=inp, outputs=z_p)
        
        ploss = [pose_loss()] * len(z_p)
        self.model.compile(loss=ploss, optimizer=RMSprop(lr=self.start_lr))
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
    
        # z_a = partial_model.output # 16 x 16 x 1024
        # assert z_a.shape.as_list() == [None, 16, 16, 1024], 'wrong shape for z_a %s' % str(z_a.shape.as_list())
        
        # return z_a
    
    def build_pose_model(self, input_shape):
        '''
        reception / stacked hourglass
        input: 256 x 256 x 3
        output: [(n_joints, dim + 1) * n_blocks, (16, 16, 1024)]
        '''
        return PoseModel(input_shape, self.dim, self.n_joints, self.n_blocks, self.reception_kernel_size).model
    
    def check_pose_output(self, pose_outputs):
        '''
        pose_outputs should be a list of poses + z_p
        '''
        assert len(pose_outputs) == self.n_blocks + 1
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
        output: 16 x 16 x 2048
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
    
    def shuffle(self, z_a, z_p):
        '''
        concat z_a and z_p (as self.concat does) but mixing z_a and z_p so they don't come from the same image
        input:
            - z_p: 16 x 16 x 1024
            - z_a: 16 x 16 x 1024
        output: 16 x 16 x 2048
        '''
        shuffled_z_p = Lambda(lambda x: tf.random.shuffle(x))(z_p)
        concat = concatenate([z_a, shuffled_z_p])
        return concat
        

