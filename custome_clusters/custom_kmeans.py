from sklearn.base import BaseEstimator, ClusterMixin
import random
import pandas as pd
import numpy as np
import copy

def get_distances(point: list, centers: list):
    distances = []
    
    return distances

def run_kmeans(X: list, centers: list, cluster_iter: int):
    
    centers_old = np.zeros(centers.shape)

    labels = np.zeros(len(X))
    
    for iter_n in range(cluster_iter):
        for i in range(len(X)):
            distances = get_distances(X, centers)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        
        centers_old = copy.deepcopy(centers)
        labels_old = copy.deepcopy(labels)
        
        for i in range(self.n_clusters):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        
        if np.array_equal(labels_old, labels):
            if verbose:
                print(f"Converged at iteration {iter_n}: strict convergence.")
            strict_convergence = True
            break
        else:
            center_shift_tot = get_center_shift(centers, centers_old)
            if center_shift_tot <= tol:
                if verbose:
                    print(f"Converged at iteration {iter_n}: center shift "
                          f"{center_shift_tot} within tolerance {tol}.")
                break

    return labels, centers

class Custom_Kmeans(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters: int, init:str='random', tol:float=1e-4 , cluster_iter:int=200):
        self.n_clusters = n_clusters
        self.tol = tol
        self.init = init
        self.cluster_iter = cluster_iter
             
    def init_centroids(self, X: list):
        
        if isinstance(self.init, str) and self.init == 'random':
            centers = random.permutation(X)[:self.n_clusters]
        else:
            raise Exception('Not implemented')
            
        return centers
            
    def fit(self, X, y=None, sample_weight=None):
        
        centers = self.init_centroids(X)
        labels, centers = run_kmeans(X, centers, self.cluster_iter)
        
        self.labels = labels
        self.centers = centers

        return self
    
    def predict(self, X, sample_weight=None):
        pass
        