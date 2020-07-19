import pandas as pd 
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
 

class Detector():
    
    def data_preprocessing (self, path):
        
        self.df = pd.read_csv(f'{path}')
        
        self.data = self.df[['timestamp','value']]
        
        self.df = preprocessing.scale(self.data)
        
        self.df = pd.DataFrame(self.df)
        self.df.columns = ['timestamp', 'value']
        
        return self.df

    def DBScan_processing (self):

        outlier_detection = DBSCAN( eps = 1.15, metric="euclidean", min_samples = 3, n_jobs = 1) 
        clusters = outlier_detection.fit_predict(self.df)

        return clusters
        
        
    def determine(self, clusters):
        self.df['anomaly3'] = pd.Series(clusters)
        
        Anomaly = self.df[(self.df.anomaly3 == -1)].index.tolist()
        
        for i in Anomaly:
            self.df.iloc[i, 2] = 1
        
        return self.df['anomaly3'].tolist()
    
    def fit_predict(self, path):
        self.data_preprocessing(path)
        clusters = self.DBScan_processing()
        result = self.determine(clusters)
        return result
       







