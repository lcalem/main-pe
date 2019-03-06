class Encoder(object):
    
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.start_lr = 0.001
        
    def stem(self, inp):
        '''
        common first stem
        '''
        print(inp.shape)
        stem_input = Input(shape=inp.shape[1:]) # 256 x 256 x 3

        x = layers.conv_bn_act(stem_input, 32, (3, 3), strides=(2, 2))
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

        x = layers.sepconv_residual(x, 3*192, name='sepconv1')

        model = Model(stem_input, x, name='stem')
        x = model(inp)
        
        return x
    
    def pose_model(self, inp):
        stem_out = self.stem(inp)
        
        out = stem_out
        
        return out
    
    def appearance_model(self, inp):
        out = ResNet50(inp)
        return out
    
    def build(self):
        '''
        Input: 256 x 256 x 3 image
        Outputs: 
            - pose tensor
            - reconstructed image
        
        1. E_p is the encoder for the pose estimation
        2. E_a is the encoder for the appearance
        3. concat z_a and z_p to form the input of the decoder
        4. decode into an image
        '''
        inp = Input(shape=self.input_shape)
        
        # 1. E_p
        z_p, pred_pose = self.pose_model(inp)
        
        # 2. E_a
        z_a = self.appearance_model(inp)
        
        # 3. reconstruction base
        concat = self.prepare_concat(z_p, z_a)
        
        # 4. decoding
        rec_img = self.decoder(concat)
        
        outputs = [pred_pose, rec_img]
        self.model = Model(inputs=inp, outputs=outputs)
        
        # compile it
        loss = losses.combined_loss()
        self.model.compile(loss=loss, optimizer=RMSprop(lr=self.start_lr))
        self.model.summary()
        
    def train(self, data_tr, steps_per_epoch):
        callbacks = []
        callbacks.append(SaveModel(weights_path))
        callbacks.append(mpii_callback)
        # callbacks.append(h36m_callback)

        model.fit_generator(
            data_tr,
            steps_per_epoch=steps_per_epoch,
            epochs=60,
            callbacks=callbacks,
            workers=8,
            initial_epoch=0)
        
        
class OldAppearanceModel(object):
    '''
    Only autoencoding z_a for now
    '''
    
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.start_lr = 0.001
        
    def decoder(self):
        pass
    
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
        
    def train(self, data_tr, steps_per_epoch, model_folder):
        weights_file = os.path.join(model_folder, 'weights_mpii_{epoch:03d}.h5')
        
        cb_list = []
        cb_list.append(callbacks.SaveModel(weights_file))
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        # callbacks.append(eval_callback)

        self.model.fit_generator(data_tr,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=60,
                                 callbacks=cb_list,
                                 workers=4,
                                 initial_epoch=0)