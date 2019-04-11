import tensorflow as tf


def pose_regression_loss(pose_loss, visibility_weight):

    def _pose_regression_loss(y_true, y_pred):
        video = len(y_true.shape) == 4
        if video:
            # The model was time-distributed, so there is one additional dimension.
            p_true = y_true[:, :, :, 0:-1]
            p_pred = y_pred[:, :, :, 0:-1]
            v_true = y_true[:, :, :, -1]
            v_pred = y_pred[:, :, :, -1]
        else:
            p_true = y_true[:, :, 0:-1]
            p_pred = y_pred[:, :, 0:-1]
            v_true = y_true[:, :, -1]
            v_pred = y_pred[:, :, -1]

        if pose_loss == 'l1l2':
            ploss = elasticnet_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l1':
            ploss = l1_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l2':
            ploss = l2_loss_on_valid_joints(p_true, p_pred)
        elif pose_loss == 'l1l2bincross':
            ploss = elasticnet_bincross_loss_on_valid_joints(p_true, p_pred)
        else:
            raise Exception('Invalid pose_loss option ({})'.format(pose_loss))

        vloss = binary_crossentropy(v_true, v_pred)

        if video:
            # If time-distributed, average the error on video frames
            vloss = K.mean(vloss, axis=-1)
            ploss = K.mean(ploss, axis=-1)

        return ploss + visibility_weight*vloss

    return _pose_regression_loss


def elastic_bce(y_true, y_pred):
    '''
    Elasticnet binary cross entropy for pose estimation
    y_true
    y_pred: (None, 16, 2)
    '''
    idx = tf.cast(tf.math.greater(y_true, 0.), tf.float32)
    #tmp_sum = tf.math.reduce_sum(idx, axis=(-1, -2))
    #print("Shape sum %s" % tmp_sum.shape)
    #num_joints = tf.clip_by_value(tmp_sum, 1, None)
    num_joints = y_pred.get_shape().as_list()[1]

    l1 = tf.math.abs(y_pred - y_true)
    l2 = tf.math.square(y_pred - y_true)
    bc = 0.01 * tf.keras.backend.binary_crossentropy(y_true, y_pred)  # doesn't expect logits like tf does
    dummy = 0. * y_pred

    return tf.reduce_sum(tf.where(tf.cast(idx, tf.bool), l1 + l2 + bc, dummy), axis=(-1, -2)) / num_joints


def pose_loss():

    def _pose_loss(y_true, y_pred):
        # print("pose y_pred shape %s" % (str(y_pred.shape)))

        pose_loss = elastic_bce(y_true, y_pred)
        return 10 * pose_loss

    return _pose_loss


def reconstruction_loss():
    '''
    scaling factor for reconstruction loss: 0.1 to keep it on the same scale as the pose loss
    '''

    def _rec_loss(y_true, y_pred):
        print("rec y_pred shape %s" % (str(y_pred.shape)))
        num_channels = y_pred.get_shape().as_list()[-1]

        rec_loss = tf.math.reduce_sum(tf.keras.backend.square(y_pred - y_true), axis=(-1, -2)) / num_channels
        return 0.1 * rec_loss

    return _rec_loss


def vgg_loss():
    '''
    perceptual loss with VGG stages
    y_pred is the concatenation of:
        - the vgg activations for the reconstructed image 
        - the vgg activations for the ground truth image
    y_true is phony here because everything is in y_pred
    
    The loss is an MSE of these activations
    The model stacks vgg_rec_outputs (for the reconstructed image) with vgg_ori_outputs (for the input image) in that order
    '''
    def _vgg_loss(y_true, y_pred):
        print("vgg y_pred shape %s (vgg_rec_outputs %s, vgg_ori_outputs %s)" % (str(y_pred.shape), str(y_pred[:, 0, :, :, :].shape), str(y_pred[:, 1, :, :, :].shape)))

        return tf.reduce_mean(tf.math.square(y_pred[0] - y_pred[1]), axis=-1)
        # num_channels = y_pred.get_shape().as_list()[-1]
        # loss = tf.math.reduce_sum(tf.keras.backend.square(y_pred - y_true), axis=(-1, -2)) / num_channels
        # return loss
        
    return _vgg_loss


def cycle_loss():
    '''
    cycle consistency loss between z representations (should be 16 x 16 x 1024 each)
    promotes disentangling
    
    y_true is phony here because we don't need anything from the dataset
    y_pred is the concatenation of z_x and z_x' so we need to separate them to build the l2 between them
    '''
    
    def _cycle_loss(y_true, y_pred):
        print("cycle y_pred shape %s" % (str(y_pred.shape)))
        z_x = y_pred[:,:,:,:1024]
        z_x_cycle = y_pred[:,:,:,1024:]
        print("z_x shape %s, z_x_cycle shape %s" % (str(z_x.shape), str(z_x_cycle.shape)))
        l2 = tf.math.square(z_x - z_x_cycle)
        return l2
    
    return _cycle_loss
        

def noop_loss():
    '''
    returns 0 no matter what
    '''
    
    def _noop_loss(y_true, y_pred):
        return tf.math.square(0.0)
    
    return _noop_loss
        