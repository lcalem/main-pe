import time

from tensorflow.keras import Model, Input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import RMSprop

from model.losses import pose_loss
from model.losses import reconstruction_loss

from model.networks import BaseModel
from model.networks.old_decoder import DecoderModel
from model.networks.old_pose import PoseModel


class MultiBranchModel(BaseModel):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    One common decoder to recreate the image
    '''

    def __init__(self, dim, n_joints=16, nb_pose_blocks=8, reception_kernel_size=(5, 5)):
        assert dim in [2, 3], 'Cannot work outside of 2D or 3D'
        
        self.dim = dim
        self.n_joints = n_joints
        self.n_blocks = nb_pose_blocks
        self.reception_kernel_size = reception_kernel_size
        
        self.verbose = True

        BaseModel.__init__(self)

    def build(self):
        # self.appearance_model = self.build_appearance_model(self.input_shape)
        self.pose_model = self.build_pose_model(self.input_shape)
        print("pose model summary")
        self.pose_model.summary()
        self.decoder_model = self.build_decoder_model((8, 8, 2048))  # i.e. 2048 for the regular model
        
        inp = Input(shape=self.input_shape)

        # encoders
        z_a = self.appearance_encoder(inp)
        # z_a = self.appearance_model(inp)
        z_p = self.pose_model(inp)

        print(type(z_a), type(z_p))
        print("Shape z_a HELLO %s" % str(z_a.shape))
        print("Shape z_p %s" % str(z_p.shape))

        # decoder
        concat = self.concat(z_a, z_p)
        print("Shape concat %s" % str(concat.shape))
        i_hat = self.decoder_model(concat)

        outputs = [i_hat, z_p]
        # outputs.extend(z_p)
        self.model = Model(inputs=inp, outputs=outputs)
        print("Outputs shape %s" % self.model.output_shape)

        # ploss = [pose_loss()] * len(z_p)
        ploss = [pose_loss()] * self.n_blocks
        losses = [reconstruction_loss()]
        losses.extend(ploss)
        
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
        
    def appearance_encoder(self, inp):
        '''
        resnet50 for now
        input: 256 x 256 x 3
        output: 8 x 8 x 2048
        '''
        enc_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)

        z_a = enc_model.output   # 8 x 8 x 2048
        return z_a
    
    def build_appearance_model(self, input_shape):
        '''
        resnet50 for now
        input: 256 x 256 x 3
        output: 16 x 16 x 1024
        '''
        enc_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        output_layer = enc_model.layers[-33]  # index of the 16 x 16 x za_depth activation we want, before the last resnet block
        assert output_layer.name.startswith('activation')
        
        partial_model = Model(inputs=enc_model.inputs, outputs=output_layer.output, name='appearance_model')
        return partial_model
    
    def build_decoder_model(self, input_shape):
        '''
        from concatenated representations to image reconstruction
        input: 16 x 16 x 1024 (z_a)
        output: 256 x 256 x 3
        '''
        return DecoderModel(input_shape=input_shape).model
    
    def build_pose_model(self, input_shape, pose_only=False):
        '''
        reception / stacked hourglass
        input: 256 x 256 x 3
        output: [(n_joints, dim + 1) * n_blocks, (16, 16, zp_depth)]   (zp_depth = 1024 or 128)
        '''
        return PoseModel(input_shape, self.dim, self.n_joints, self.n_blocks, self.reception_kernel_size).model
    
    def concat(self, z_a, z_p):
        '''
        concat pose and appearance representations before decoding
        input:
            - z_p:
            - z_a: 8 x 8 x 2048
        output:

        TODO: This is where the real work should happen
        '''
        return z_a
