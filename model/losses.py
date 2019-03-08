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
    print("Shape %s" % idx.shape)
    #tmp_sum = tf.math.reduce_sum(idx, axis=(-1, -2))
    #print("Shape sum %s" % tmp_sum.shape)
    #num_joints = tf.clip_by_value(tmp_sum, 1, None)
    num_joints = y_pred.get_shape().as_list()[1]
    print("Num joints %s" % num_joints)

    l1 = tf.math.abs(y_pred - y_true)
    l2 = tf.math.square(y_pred - y_true)
    bc = 0.01 * tf.keras.backend.binary_crossentropy(y_true, y_pred)  # doesn't expect logits like tf does
    dummy = 0. * y_pred

    return tf.reduce_sum(tf.where(tf.cast(idx, tf.bool), l1 + l2 + bc, dummy), axis=(-1, -2)) / num_joints


def pose_loss():

    def _pose_loss(y_true, y_pred):
        print("pose y_true shape %s" % (str(y_true.shape)))
        print("pose y_pred shape %s" % (str(y_pred.shape)))

        pose_loss = elastic_bce(y_true, y_pred)
        return pose_loss

    return _pose_loss


def reconstruction_loss():

    def _rec_loss(y_true, y_pred):
        print("rec y_true shape %s" % (str(y_true.shape)))
        print("rec y_pred shape %s" % (str(y_pred.shape)))
        num_joints = y_pred.get_shape().as_list()[-1]
        print("Num joints: %s" % num_joints)

        rec_loss = tf.math.reduce_sum(tf.keras.backend.square(y_pred - y_true), axis=(-1, -2)) / num_joints
        return rec_loss

    return _rec_loss
