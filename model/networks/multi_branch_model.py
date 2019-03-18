import time

from tensorflow.keras import Model, Input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop

from model.losses import pose_loss
from model.losses import reconstruction_loss

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
        inp = Input(shape=self.input_shape)

        # encoders
        time_1 = time.time()
        z_a = self.appearance_encoder(inp)
        time_2 = time.time()
        pose_outputs = self.pose_encoder(inp)
        time_3 = time.time()

        print("Build E_a %s, build E_p %s" % (time_2 - time_1, time_3 - time_2))
        
        poses, z_p = self.check_pose_output(pose_outputs)
        print("Shape z_a %s, shape z_p %s" % (str(z_a.shape), str(z_p.shape)))

        # decoder
        concat = self.concat(z_a, z_p)
        print("Shape concat %s" % str(concat.shape))
        assert concat.shape.as_list() == [None, 16, 16, 2048], 'wrong concat shape %s' % str(concat.shape)
        i_hat = self.decoder(concat)

        outputs = [i_hat]
        outputs.extend(poses)
        self.model = Model(inputs=inp, outputs=outputs)
        print("Outputs shape %s" % self.model.output_shape)

        ploss = [pose_loss()] * len(poses)
        losses = [reconstruction_loss()]
        losses.extend(ploss)
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

    def appearance_encoder(self, inp):
        '''
        resnet50 for now
        input: 256 x 256 x 3
        output: 16 x 16 x 1024
        '''
        enc_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)
        output_layer = enc_model.layers[-33]  # index of the 16 x 16 x 1024 activation we want, before the last resnet block
        assert output_layer.name.startswith('activation')
        
        partial_model = Model(inputs=enc_model.inputs, outputs=output_layer.output)
        z_a = partial_model.output # 16 x 16 x 1024
        assert z_a.shape.as_list() == [None, 16, 16, 1024], 'wrong shape for z_a %s' % str(z_a.shape.as_list())
        
        return z_a

    def pose_encoder(self, inp):
        '''
        reception / stacked hourglass
        input: 256 x 256 x 3
        output: [(n_joints, dim + 1) * n_blocks, (16, 16, 1024)]
        '''
        pose_model = PoseModel(inp, self.dim, self.n_joints, self.n_blocks, self.reception_kernel_size).model
        out = pose_model.output

        return out
    
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
        output:

        TODO: This is where the real work should happen
        '''
        concat = concatenate([z_a, z_p])
        return concat

    def decoder(self, concat):
        '''
        from concatenated representations to image reconstruction
        input: 16 x 16 x 2048 (z_a and z_p concatenated)
        output: 256 x 256 x 3
        '''
        decoder_model = DecoderModel(input_tensor=concat).model
        out = decoder_model(concat)

        return out

