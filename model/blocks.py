import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import add
from model import layers


def sepconv_residual(x, out_size, name, kernel_size=(3, 3)):
    '''
    Separable convolution with residual 
    TODO: Schema
    '''
    shortcut_name = name + '_shortcut'
    reduce_name = name + '_reduce'

    num_filters = K.int_shape(x)[-1]
    if num_filters == out_size:
        ident = x
    else:
        ident = layers.act_conv_bn(x, out_size, (1, 1), name=shortcut_name)

    if out_size < num_filters:
        x = layers.act_conv_bn(x, out_size, (1, 1), name=reduce_name)

    x = layers.separable_act_conv_bn(x, out_size, kernel_size, name=name)
    x = add([ident, x])

    return x
