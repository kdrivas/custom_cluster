from sklearn.base import BaseEstimator, ClusterMixin
import random
import pandas as pd
import numpy as np
import copy
import types

def get_center_shift(centers, centers_old):
    shift = 0
    for center, center_old in zip(centers, centers_old):
        shift += np.linalg.norm(center - center_old)

    return shift
        
def get_distances(point:list, centers:list):
    distances = []
    for center in centers:
        distances.append(np.linalg.norm(center - point))
    return distances

def run_kmeans(X:list, centers:list, distance_function:types.LambdaType, cluster_iter:int, n_clusters:int, tol:float, verbose:bool):
    
    centers_old = np.zeros(centers.shape)

    labels = np.zeros(len(X))
    labels_old = np.zeros(len(X))
    
    for iter_n in range(cluster_iter):
        for i in range(len(X)):
            distances = distance_function(X[i], centers)
            cluster = np.argmin(distances)
            labels[i] = cluster
        
        for i in range(n_clusters):
            points = [X[j] for j in range(len(X)) if labels[j] == i]
            centers[i] = np.mean(points, axis=0)

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
                
        centers_old = copy.deepcopy(centers)
        labels_old = copy.deepcopy(labels)

    return labels, centers

class Custom_Kmeans(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters: int, init:str='random', tol:float=1e-4 , cluster_iter:int=50, verbose:bool=True, distance_function:types.LambdaType=get_distances):
        self.n_clusters = n_clusters
        self.tol = tol
        self.init = init
        self.cluster_iter = cluster_iter
        self.verbose = verbose
        self.distance_function = distance_function
             
    def init_centroids(self, X:list):
        
        if isinstance(self.init, str) and self.init == 'random':
            centers = np.random.permutation(X)[:self.n_clusters]
        else:
            raise Exception('Not implemented')
            
        return centers
            
    def fit(self, X, y=None, sample_weight=None):
        
        centers = self.init_centroids(X)
        labels, centers = run_kmeans(X, centers, self.distance_function, self.cluster_iter, self.n_clusters, self.tol, self.verbose)
        
        self.labels = labels
        self.cluster_centers_ = centers

        return self
    
    def predict(self, X, sample_weight=None):
        pass
        