from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import RMSprop

from model import layers
from model.networks import BaseModel


class AppearanceModel(BaseModel):
    '''
    Only autoencoding z_a for now
    '''

    def build(self):
        inp = Input(shape=self.input_shape)

        enc_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)

        z_a = enc_model.output   # 8 x 8 x 2048

        # decoder part
        up = layers.up(z_a)  # 16 x 16
        up = layers.conv_bn_act(up, 512, (3, 3))
        up = layers.conv_bn_act(up, 512, (3, 3))
        up = layers.conv_bn_act(up, 512, (3, 3))

        up = layers.up(up)  # 32 x 32
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

        self.model = Model(inputs=inp, outputs=i_hat)

        # loss = losses.combined_loss()
        loss = mean_squared_error

        # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        # self.model.compile(loss=loss, optimizer=RMSprop(lr=self.start_lr), options=run_opts)
        self.model.compile(loss=loss, optimizer=RMSprop(lr=self.start_lr))
        self.model.summary()

    def evaluation(self, data_eval):
        raise NotImplementedError
