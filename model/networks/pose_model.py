import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import Activation, GlobalMaxPooling1D, GlobalMaxPooling2D, Lambda, MaxPooling2D, UpSampling2D

from model import blocks
from model import layers


DEPTH_MAPS = 16


class PoseModel(object):

    def __init__(self, input_shape, dim, n_joints, n_blocks, kernel_size, pose_only=False, verbose=True):
        self.dim = dim
        
        self.n_joints = n_joints
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.pose_only = pose_only
        self.verbose = verbose

        if dim == 2:
            self.n_heatmaps = self.n_joints
        elif dim == 3:
            self.depth_maps = 16
            self.n_heatmaps = self.depth_maps * self.n_joints
        else:
            raise Exception('Dim can only be 2 or 3 (was %s)' % dim)

        self.build(input_shape)

    @property
    def model(self):
        return self._model
    
    def log(self, msg):
        if self.verbose:
            print(msg)

    def build(self, input_shape):
        '''
        1. stem
        2. stacking the blocks
        
        Shapes:
        input: 256 x 256 x 3
        outputs: [(n_joints, dim + 1) * n_blocks, (16 x 16 x 1024)]
        
        - last element of outputs is z_p (16 x 16 x 1024)
        - remaining elements are the pose regression (one per block), size (n_joints, dim + 1)
            - n_joints: 16 (MPII) or 17 (Human 3.6M)
            - dim + 1: dimension (2 or 3) + 1 for visibility prediction
            -> (16, 3) or (17, 4)
        '''

        inp = Input(shape=input_shape)
        outputs = list()
        
        x = self.stem(inp)

        # static layers
        num_rows, num_cols, num_filters = x.get_shape().as_list()[1:]
        # print("num rows %s, num cols %s, num filters %s" % (num_rows, num_cols, num_filters))
        pose_input_shape = (num_rows, num_cols, self.n_joints)   # (32, 32, 17) like 
        self.pose_softargmax_model = self.build_softargmax_model(pose_input_shape)
        self.joint_visibility_model = self.build_visibility_model(pose_input_shape)
        self.pose_depth_model = self.build_depth_model()   # will be None for dim 2

        # hourglass blocks
        for i_block in range(self.n_blocks):

            block_shape = x.get_shape().as_list()[1:]
            x = self.reception_block(x, name='rBlock%d' % (i_block + 1))

            identity_map = x
            x = self.sepconv_block(x, name='SepConv%d' % (i_block + 1))
            h = self.pose_block(x, name='RegMap%d' % (i_block + 1))

            pose, visible = self.pose_regression(h, name='PoseReg%s' % (i_block + 1))
            pose_vis = concatenate([pose, visible], axis=-1)
            self.log("pose shape %s, vis shape %s, concat shape %s" % (str(pose.shape), str(visible.shape), str(pose_vis.shape)))

            outputs.append(pose_vis)

            # if i_block < self.n_blocks - 1:
            h = self.fremap_block(h, block_shape[-1], name='fReMap%d' % (i_block + 1))
            x = add([identity_map, x, h])
            
        # z_p from last block
        if not self.pose_only:
            self.log("Last H shape %s" % str(h))
            z_p = MaxPooling2D((2, 2))(h)
            z_p = layers.act_conv_bn(z_p, 1024, (1, 1))
            outputs.append(z_p)

        self._model = Model(inputs=inp, outputs=outputs)

    def stem(self, inp):
        '''
        inception v4 stem

        input: 256 x 256 x 3
        output: 32 x 32 x 576
        '''
        xi = Input(shape=inp.get_shape().as_list()[1:]) # 256 x 256 x 3

        x = layers.conv_bn_act(xi, 32, (3, 3), strides=(2, 2))
        x = layers.conv_bn_act(x, 32, (3, 3))
        x = layers.conv_bn_act(x, 64, (3, 3))

        a = layers.conv_bn_act(x, 96, (3, 3), strides=(2, 2))
        b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = concatenate([a, b])

        a = layers.conv_bn_act(x, 64, (1, 1))
        a = layers.conv_bn(a, 96, (3, 3))
        b = layers.conv_bn_act(x, 64, (1, 1))
        b = layers.conv_bn_act(b, 64, (5, 1))
        b = layers.conv_bn_act(b, 64, (1, 5))
        b = layers.conv_bn(b, 96, (3, 3))
        x = concatenate([a, b])

        a = layers.act_conv_bn(x, 192, (3, 3), strides=(2, 2))
        b = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = concatenate([a, b])

        x = blocks.sepconv_residual(x, 3 * 192, name='sepconv1')

        model = Model(xi, x, name='Stem')
        x = model(inp)

        return x

    def build_softargmax_model(self, input_shape):
        '''
        Static model for soft argmax
        '''

        inp = Input(shape=input_shape)
        x = layers.act_channel_softmax(inp)

        x_x = layers.lin_interpolation_2d(x, dim=0)
        x_y = layers.lin_interpolation_2d(x, dim=1)
        pose = concatenate([x_x, x_y])

        model = Model(inputs=inp, outputs=pose)
        model.trainable = False

        return model

    def build_visibility_model(self, input_shape):
        '''
        Static model for joint visibility
        '''
        num_rows, num_cols = input_shape[0:2]
        inp = Input(shape=input_shape)

        x = MaxPooling2D((num_rows, num_cols))(inp)
        x = Activation('sigmoid')(x)

        x = Lambda(lambda x: tf.squeeze(x, axis=1))(x)
        x = Lambda(lambda x: tf.squeeze(x, axis=1))(x)
        x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)

        model = Model(inputs=inp, outputs=x)

        return model
    
    def build_depth_model(self):
        '''
        Static model (1D soft argmax on z axis)
        Only for dim 3
        '''
        input_shape = (self.depth_maps, self.n_joints)
        name_sm = 'zSAM_softmax'

        inp = Input(shape=input_shape)
        x = layers.act_depth_softmax(inp, name=name_sm)
        x = layers.lin_interpolation_1d(x)

        model = Model(inputs=inp, outputs=x, name=name_sm)
        model.trainable = False

        return model

    def reception_block(self, inp, name):
        '''
        each pose block starts with a reception block
        it is u-shaped and relies on separable convolutions

        inp ------------------------- a (SR 576) -------------------- + -- out
          |                                                           |
          |                                                           |
          MP --- C 288 -- SR 288 ---- b (SR 288) ---- + -- SR 576 -- US
                            |                         |
                            |                         |
                            MP --- SR -- SR -- SR --- US     <- all 288 channels


        SR: Sepconv Residual (all 5x5)
        C: Conv (1x1)
        MP: Max Pooling (2x2 with stride 2x2)
        US: UpSampling (2x2)

        input: 32 x 32 x 576
        output: 32 x 32 x 576
        '''
        ksize = self.kernel_size

        input_shape = inp.get_shape().as_list()[1:]
        size = int(input_shape[-1])

        # first branch
        xi = Input(shape=input_shape)
        a = blocks.sepconv_residual(xi, size, name='sepconv_l1', kernel_size=ksize)

        # second branch
        low1 = MaxPooling2D((2, 2))(xi)
        low1 = layers.act_conv_bn(low1, int(size/2), (1, 1))
        low1 = blocks.sepconv_residual(low1, int(size/2), name='sepconv_l2_1', kernel_size=ksize)
        b = blocks.sepconv_residual(low1, int(size/2), name='sepconv_l2_2', kernel_size=ksize)

        # third branch
        c = MaxPooling2D((2, 2))(low1)
        c = blocks.sepconv_residual(c, int(size/2), name='sepconv_l3_1', kernel_size=ksize)
        c = blocks.sepconv_residual(c, int(size/2), name='sepconv_l3_2', kernel_size=ksize)
        c = blocks.sepconv_residual(c, int(size/2), name='sepconv_l3_3', kernel_size=ksize)
        c = UpSampling2D((2, 2))(c)

        # merge second and third branches
        b = add([b, c])
        b = blocks.sepconv_residual(b, size, name='sepconv_l2_3', kernel_size=ksize)
        b = UpSampling2D((2, 2))(b)

        # merge first and second branches
        x = add([a, b])
        model = Model(inputs=xi, outputs=x, name=name)

        return model(inp)

    def sepconv_block(self, inp, name):
        '''
        Separable convolution
        '''
        input_shape = inp.get_shape().as_list()[1:]

        xi = Input(shape=input_shape)
        x = layers.separable_act_conv_bn(xi, input_shape[-1], self.kernel_size)

        model = Model(inputs=xi, outputs=x, name=name)

        return model(inp)

    def pose_block(self, inp, name):
        '''
        input: 32 x 32 x 576
        output: 32 x 32 x 16 (number of heatmaps)
        '''
        input_shape = inp.get_shape().as_list()[1:]

        xi = Input(shape=input_shape)
        x = layers.act_conv(xi, self.n_heatmaps, (1, 1))

        model = Model(inputs=xi, outputs=x, name=name)

        return model(inp)
    
    def pose_regression(self, heatmaps, name):
        if self.dim == 2:
            return self.pose_regression_2d(heatmaps, name)
        elif self.dim == 3:
            return self.pose_regression_3d(heatmaps, name)
        else:
            raise Exception('This should not happen so far in the model')

    def pose_regression_2d(self, heatmaps, name):
        '''
        soft argmax to get the pose from the heatmaps
        joint prob model to get the joint visibility probability

        input: 32 x 32 x 16 (number of joints)
        output:
        - pose (None, 16, 2)
        - visibility (None, 16, 1)
        '''
        pose = self.pose_softargmax_model(heatmaps)
        visibility = self.joint_visibility_model(heatmaps)

        return pose, visibility
    
    def pose_regression_3d(self, heatmaps, name):
        '''
        
        input: 32 x 32 x 17*16 (njoints (17) times depth maps)
        output:
        - pose (None, 17, 3)
        - visibility (None, 17, 1)
        '''
        assert heatmaps.get_shape().as_list()[-1] == self.depth_maps * self.n_joints  # the number of heatmaps

        def _reshape_heatmaps(x):
            x = tf.expand_dims(x, axis=-1)
            shape = x.get_shape().as_list()
            x = tf.reshape(x, (-1, shape[1], shape[2], self.depth_maps, self.n_joints))

            return x

        # separate heatmaps into 2D x-y heatmaps and depth z heatmaps
        h = Lambda(_reshape_heatmaps)(heatmaps)
        hxy = Lambda(lambda x: tf.reduce_mean(x, axis=3))(h)
        hz = Lambda(lambda x: tf.reduce_mean(x, axis=(1, 2)))(h)
        # hxy = Lambda(lambda x: K.max(x, axis=3))(h)
        # hz = Lambda(lambda x: K.max(x, axis=(1, 2)))(h)

        # hxy_s = act_channel_softmax(hxy)
        # hz_s = act_depth_softmax(hz)

        # x-y are taken in the same pose soft argmax model as regular heatmaps
        pose_xy = self.pose_softargmax_model(hxy)
        pose_z = self.pose_depth_model(hz)
        pose = concatenate([pose_xy, pose_z])

        vxy = GlobalMaxPooling2D()(hxy)
        vz = GlobalMaxPooling1D()(hz)
        v = add([vxy, vz])
        v = Lambda(lambda x: tf.expand_dims(x, axis=-1))(v)
        visible = Activation('sigmoid')(v)

        return pose, visible

    def fremap_block(self, inp, num_filters, name=None):
        input_shape = inp.get_shape().as_list()[1:]

        xi = Input(shape=input_shape)
        x = layers.act_conv_bn(xi, num_filters, (1, 1))

        model = Model(inputs=xi, outputs=x, name=name)

        return model(inp)
