
from tensorflow.keras import Model, Input

from model import layers


class DecoderModel(object):

    def __init__(self, input_shape):

        self.build(input_shape)

    @property
    def model(self):
        return self._model

    def build(self, input_shape):
        z_a = Input(shape=input_shape)
        # z_a = Input(shape=inp.get_shape().as_list()[1:])  # for now, only the z_a part (8 x 8 x 2048)

        up = layers.up(z_a)  # 32 x 32
        up = layers.conv_bn_act(up, 512, (3, 3))
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

        self._model = Model(inputs=z_a, outputs=i_hat, name='decoder')


