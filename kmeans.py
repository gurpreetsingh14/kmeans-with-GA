# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:21:29 2022


https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

@author: micho
"""

import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

def euclidean(point, data):
    euc = np.sqrt(np.sum((point-data)**2, axis = 1))
    
    return euc

# def euclidean(point, data): # both point and data are arrays
#     l=[]
#     for row in data:
#         s = (point[0]-row[0])**2 + (point[1]-row[1])**2
#         euc = np.sqrt(s)
#         l.append(euc)
#     return np.array(l)

class KMeans:
    def __init__(self, n_clusters, max_iter = 400):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def fit(self, X_train, verbose = False, p = False):
        # randomly select the centroid start points according to a uniform distribution
        min_, max_ = np.min(X_train, axis = 0), np.max(X_train, axis = 0)
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            classified_points = [[] for _ in range(self.n_clusters)]
            if verbose: print("Iteration",iteration)
            for x in X_train:
                # print('X:',x)
                # print('Centroids:',np.array(self.centroids))
                dist = euclidean(x, self.centroids)
                # print('Distances',dist)
                centroid_id = np.argmin(dist)
                # print('Centroid ID',centroid_id)
                classified_points[centroid_id].append(x)
                
            # updated centroids
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis = 0) for cluster in classified_points]
                
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i] # don't update
            if verbose: print("Previous centroids:",prev_centroids)
            if verbose: print("New centroids:", self.centroids)
            if verbose: print("==============================")
            iteration +=1
            
    def evaluate(self, X):
        centroids = []
        centroid_ids = []
        for x in X:
            dist = euclidean(x, self.centroids)
            centroid_id = np.argmin(dist)
            centroids.append(self.centroids[centroid_id])
            centroid_ids.append(centroid_id)
            
        return centroids, centroid_ids
    
    def to__solution(self):
        l = []
        for el in self.centroids:
            l.append(el[0])
            l.append(el[1])
        return np.array(l) 
