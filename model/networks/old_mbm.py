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
        
        self.zp_depth = 1024

        BaseModel.__init__(self)

    def build(self):
        self.build_everything()
        
        inp = Input(shape=self.input_shape)
        self.log("Input shape %s" % str(inp.shape))
        
        z_a = self.appearance_model(inp)
        z_p = self.pose_model(inp)
        # pose_outputs = self.pose_model(inp)
        # poses, z_p = self.check_pose_output(pose_outputs)
        
        # self.log("Shape z_a %s, shape z_p %s" % (str(z_a.shape), str(z_p.shape)))

        # decoder
        concat = self.concat(z_a, z_p)
        print("Shape concat %s" % str(concat.shape))
        i_hat = self.decoder_model(concat)

        outputs = [i_hat]
        # outputs.extend(poses)
        outputs.extend(z_p)
        self.model = Model(inputs=inp, outputs=outputs)
        print("Outputs shape %s" % self.model.output_shape)

        ploss = [pose_loss()] * len(z_p)
        losses = [reconstruction_loss()]
        losses.extend(ploss)
        # loss = mean_squared_error
        self.model.compile(loss=losses, optimizer=RMSprop(lr=self.start_lr))
        self.model.summary()
        
    def build_everything(self):
        time_1 = time.time()
        self.appearance_model = self.build_appearance_model(self.input_shape)
        time_2 = time.time()
        self.pose_model = self.build_pose_model(self.input_shape)
        time_3 = time.time()
        self.decoder_model = self.build_decoder_model((16, 16, 1024))  # i.e. 2048 for the regular model
        time_4 = time.time()
        
        self.log("Build E_a %s, build E_p %s, decoder D %s" % (time_2 - time_1, time_3 - time_2, time_4 - time_3))
        
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
        output_layer = enc_model.layers[-33]  # index of the 16 x 16 x za_depth activation we want, before the last resnet block
        assert output_layer.name.startswith('activation')
        
        partial_model = Model(inputs=enc_model.inputs, outputs=output_layer.output, name='appearance_model')
        return partial_model
    
    def build_pose_model(self, input_shape, pose_only=False):
        '''
        reception / stacked hourglass
        input: 256 x 256 x 3
        output: [(n_joints, dim + 1) * n_blocks, (16, 16, zp_depth)]   (zp_depth = 1024 or 128)
        '''
        return PoseModel(input_shape, self.dim, self.n_joints, self.n_blocks, self.reception_kernel_size).model
    
    def build_decoder_model(self, input_shape):
        '''
        from concatenated representations to image reconstruction
        input: 16 x 16 x 1024 (z_a and z_p concatenated)    [or 256]
        output: 256 x 256 x 3
        '''
        return DecoderModel(input_shape=input_shape).model

    def concat(self, z_a, z_p):
        '''
        concat pose and appearance representations before decoding
        input:
            - z_p:
            - z_a: 16 x 16 x 1024
        output:

        '''
        return z_a
    
    def check_pose_output(self, pose_outputs):
        '''
        pose_outputs should be a list of poses + z_p
        '''
        assert len(pose_outputs) == self.n_blocks + 1  # + 1 for the z_p (16 x 16 x zp_depth) representation
        poses = pose_outputs[:-1]
        z_p = pose_outputs[-1]
        
        pose_shapes = [pose.shape.as_list() for pose in poses]
        assert all([shape == [None, self.n_joints, self.dim + 1] for shape in pose_shapes]), 'pose shapes are weird %s' % str(pose_shapes)
        assert z_p.shape.as_list() == [None, 16, 16, self.zp_depth], 'z_p shape not as expected %s' % str(z_p.shape.as_list())
        
        return poses, z_p