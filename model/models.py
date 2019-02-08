import os

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from model import layers
from model import losses
from model import config
from model import callbacks
from model.utils import pose_format


class BaseModel(object):
    '''
    Base class for different models
    '''
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.start_lr = 0.001
        
    def load(self, checkpoint_path):
        self.model = load_model(checkpoint_path)
        
    def build(self):
        raise NotImplementedError
        
    def train(self, data_tr, steps_per_epoch, model_folder, n_epochs):
        weights_file = os.path.join(model_folder, 'appearance_mpii_{epoch:03d}.h5')
        
        cb_list = []
        cb_list.append(callbacks.SaveModel(weights_file))
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        # callbacks.append(eval_callback)

        self.model.fit_generator(data_tr,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=n_epochs,
                                 callbacks=cb_list,
                                 workers=4,
                                 initial_epoch=0)
        
    def evaluation(self, data_eval):
        pass
    
    
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
        

class MultiBranchModel(BaseModel):
    '''
    2 branch model :
    - appearance (z_a)
    - pose (z_p)
    '''
    
    def build(self):
        inp = Input(shape=self.input_shape)
        
        z_a = self.appearance_encoder(inp)
        z_p = self.pose_encoder(inp)
        
        concat = self.concat(z_a, z_p)
        
        i_hat = self.decoder(concat)
        
        self.model = Model(inputs=inp, outputs=[z_p, i_hat])
       
        loss = multi_loss
        self.model.compil(loss=loss, optimizer=RMSProp(lr=self.start_lr))
        self.model.summary()
        
    def appearance_encoder(self, inp):
        pass
        
        