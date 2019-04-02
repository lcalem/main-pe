
from tensorflow.keras import Model, Input

# from model import layers
import importlib.machinery
layers = importlib.machinery.SourceFileLoader('layers', '/home/caleml/pe_experiments/exp_20190322_1942_hybrid_h36m__1b_bs16/model_src/model/layers.py').load_module()

class DecoderModel(object):

    def __init__(self, input_shape):

        self.build(input_shape)

    @property
    def model(self):
        return self._model

    def build(self, input_shape):
        '''
        input: concat of z_a and z_p -> 16 x 16 x 2048
        output: reconstructed image 256 x 256 x 3
        '''
        concat = Input(shape=input_shape)

        up = layers.up(concat)  # 32 x 32
        up = layers.conv_bn_act(up, 1024, (3, 3))
        up = layers.conv_bn_act(up, 512, (3, 3))
        up = layers.conv_bn_act(up, 256, (3, 3))

        up = layers.up(up)  # 64 x 64
        up = layers.conv_bn_act(up, 256, (3, 3))
        up = layers.conv_bn_act(up, 256, (3, 3))
        up = layers.conv_bn_act(up, 128, (3, 3))

        up = layers.up(up)  # 128 x 128
        up = layers.conv_bn_act(up, 128, (3, 3))
        up = layers.conv_bn_act(up, 64, (3, 3))

        up = layers.up(up)  # 256 x 256
        up = layers.conv_bn_act(up, 3, (3, 3))
        up = layers.conv_bn(up, 3, (1, 1))   # 3 channels, output shape of this should be (None, 3, 256, 256)

        # TODO: should we permute here or have the input formatted with channels first?
        # perm = Permute((1, 2))(up)
        # i_hat = Permute((2, 3))(perm)
        i_hat = up

        self._model = Model(inputs=concat, outputs=i_hat, name='decoder')


