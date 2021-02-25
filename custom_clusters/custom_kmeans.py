from sklearn.base import BaseEstimator, ClusterMixin
import random
import pandas as pd
import numpy as np
import copy
import types
from sklearn.utils.extmath import row_norms
from sklearn.cluster import kmeans_plusplus

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

def run_kmeans(X:list, attention_start:list, attention_finish:list, centers:list, distance_function:types.LambdaType, cluster_iter:int, n_clusters:int, tol:float, optimize_iterations:bool, time_diff:float=5, verbose:bool=True):
    
    centers_old = np.zeros(centers.shape)

    labels = np.zeros(len(X), dtype=int)
    labels_old = np.zeros(len(X))
    distance_to_center = np.zeros(len(X))
    
    attended_flag = np.zeros(len(X), dtype=int)
    attended_by = [-1] * len(X)
    attended_by_old = [-1] * len(X)
    
    cant = 0
    for iter_n in range(cluster_iter):
        finish_time = np.zeros(n_clusters, dtype='<M8[ns]')

        for i in range(len(X)):
            if optimize_iterations:
                avaiability = np.array([j for j in range(n_clusters) if ((attention_start[i] - finish_time[j]) / np.timedelta64(1, 's')) >= 0])
                cluster = -1
                flag_distance = False
                
                if len(avaiability):
                    cant += 1
                    if attended_by[i] == -1 or attended_by[i] not in avaiability:
                        distances = distance_function(X[i], centers[avaiability])
                        flag_distance = True
                    else: 
                        avaiability_aux = avaiability[avaiability != attended_by[i]]
                        first_distance = distance_function(X[i], [centers[attended_by[i]]])
                        if (first_distance[0]  + time_diff) > distance_to_center[i]:
                            #Reduction request
                            distances = distance_function(X[i], centers[avaiability_aux]) + first_distance
                            avaiability = np.append(avaiability_aux, attended_by[i])
                            flag_distance = True
                    
                    #flag_distance = True
                    #distances = distance_function(X[i], centers[avaiability])
                    
                    if flag_distance:
                        sorted_distances = np.argsort(distances)

                        for distance_id in sorted_distances:
                            if (finish_time[avaiability[distance_id]] and ((attention_start[i] - finish_time[avaiability[distance_id]]) / np.timedelta64(1, 's')) >= 0) or not finish_time[avaiability[distance_id]]:
                                cluster = distance_id
                                break
                
                if flag_distance:
                    distance_to_center[i] = distances[cluster]
                    attended_flag[i] = 1
                    attended_by[i] = avaiability[cluster]
                    finish_time[cluster] = attention_finish[i]
                elif not flag_distance and len(avaiability):
                    finish_time[cluster] = attention_finish[i]
                else:
                    distance_to_center[i] = distances[cluster]
                    attended_flag[i] = 1
                    attended_by[i] = cluster
                    
            else:
                raise Exception('Not implemented')
                #distances = distance_function(X[i], centers)
                #cluster = np.argmin(distances)
                #labels[i] = cluster

        for i in range(n_clusters):
            points = [X[j] for j in range(len(X)) if attended_by[j] == i]
            if len(points):
                centers[i] = np.mean(points, axis=0)

        if np.array_equal(attended_by_old, attended_by):
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
        attended_by_old = copy.deepcopy(attended_by)
    
    return attended_by, centers

class FacilityEstimator(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters: int, init:str='kmeans++', tol:float=1e-4 , cluster_iter:int=50, verbose:bool=True, distance_function:types.LambdaType=get_distances, optimize_iterations:bool=True, random_state:int=0):
        self.n_clusters = n_clusters
        self.tol = tol
        self.init = init
        self.cluster_iter = cluster_iter
        self.verbose = verbose
        self.distance_function = distance_function
        self.random_state = random_state
        self.optimize_iterations = optimize_iterations
             
    def _init_centroids(self, X:list, x_squared_norms:list):
        
        if isinstance(self.init, str) and self.init == 'random':
            centers = np.random.permutation(X)[:self.n_clusters]
        elif isinstance(self.init, str) and self.init == 'kmeans++':
            centers, _ = kmeans_plusplus(X, self.n_clusters,
                                          random_state=self.random_state,
                                          x_squared_norms=x_squared_norms)
        else:
            raise Exception('Not implemented')
            
        return centers
            
    def fit(self, X, attention_hour, attention_times, y=None, sample_weight=None):
        
        x_squared_norms = row_norms(X, squared=True)
        
        centers = self._init_centroids(X, x_squared_norms)
        labels, centers = run_kmeans(X, attention_hour, attention_times, centers, self.distance_function, self.cluster_iter, self.n_clusters, self.tol, self.optimize_iterations, self.verbose)
        
        self.labels = labels
        self.cluster_centers_ = centers

        return self
    
    def predict(self, X, sample_weight=None):
        pass
        