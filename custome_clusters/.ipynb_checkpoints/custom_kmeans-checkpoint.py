from sklearn.base import BaseEstimator, ClusterMixin

class Custom_Kmeans(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None, sample_weight=None):
        return self
    
    def predict(self, X, sample_weight=None):
        pass
        