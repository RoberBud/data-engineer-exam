from creme import anomaly
from creme import compose
from creme import preprocessing

class Detector:
    def __init__(self):
        
        self.model = anomaly.HalfSpaceTrees(n_trees = 10 , height = 3 ,window_size= 25, seed=42,limits = {'ptr':[-15,32]})
            
        self.scaler = preprocessing.StandardScaler()
        
    def fit_predict(self, ptr):
        features = {'ptr': float(ptr)}
        features = self.scaler.fit_one(features).transform_one(features)
        
        self.model = self.model.fit_one(features)
       
        if self.model.score_one(features) >= 0.5:
            pred = 1
        else:
            pred = 0
        
       
        return pred
