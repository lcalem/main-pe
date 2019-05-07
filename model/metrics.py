from model import measures

    
class BaseMetric(object):
    default_name = "basemetric"

    def __init__(self, name=None):
        self.__name__ = name or self.default_name

    def __call__(self, y_true, y_pred, **kwargs):
        return self.compute_metric(y_true, y_pred, **kwargs)

    def compute_metric(self, y_true, y_pred):
        raise NotImplementedError
        
        
class MPJPEMetric(BaseMetric):
    
    def __init__(self, 
                 nb_blocks,
                 pose_only=False):
        '''
        '''
        self.nb_blocks = nb_blocks
        self.pose_only = pose_only
        
        BaseMetric.__init__(self, "mpjpe")
        
    def compute_metric(self, y_true, y_pred):
        '''
        for computing the MPJPE during training (keras metrics)

        y_true and y_pred differ from model to model!
        We must find the values in both lists that correspond to the (None, 17, 4) tensor (i.e. the pose + vis tensor)
        '''
        # assert len(y_true) == len(y_pred)
        
        print("METRIC y_true %s" % str(y_true.shape))
        scam = y_true[-1]
        rootz = y_true[-2][:,0,2]    # from pose_uvd
        afmat = y_true[-3]
        pose_w = y_true[-4]
        
        # first output is image and pose output start after if not pose_only model
        pred = y_pred if self.pose_only else y_pred[1:]
        
        measures.compute_mpjpe(self.nb_blocks,
                              pred,
                              pose_w,
                              afmat,
                              rootz,
                              scam,
                              verbose=False)
        
       