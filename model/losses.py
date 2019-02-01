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