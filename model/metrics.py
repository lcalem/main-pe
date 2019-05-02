from model import measure

    
class MPJPEMetric():
        
        
    def __call__(self, y_true, y_pred):
        '''
        for computing the MPJPE during training (keras metrics)

        y_true and y_pred differ from model to model!
        We must find the values in both lists that correspond to the (None, 17, 4) tensor (i.e. the pose + vis tensor)
        '''
        assert len(y_true) == len(y_pred)
        
        measure.mean_distance_error(y_true, y_pred)