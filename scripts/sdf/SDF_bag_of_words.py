""" 
Definition of SDFBagOfWords Class
"""
import numpy as np
#from operator import itemgetter

from sklearn.cluster import KMeans

from sdf_class import SDF
from random_functions import histogramize


class SDFBagOfWords:
    def __init__(self, num_clusters=10):
        self.Kmeans_model = None
        self.cluster_centers = None
        self.NUM_CLUSTERS = num_clusters
        self.W = 7
        self.S = 10
        self.FILTER = False
    
    def set_W(self, W):
        self.W = W
    
    def set_S(self, S):
        self.S = S
   
    def set_num_clusters(self, num_clusters):
        self.NUM_CLUSTERS = num_clusters

    def set_window_filtering(self, filtering):
        self.FILTER = filtering

    def fit(self, sdf_list, K):
        windows = np.array([])
        for file_ in sdf_list:
            #print file_
            converted = SDF(file_)
            my_windows = np.array(converted.make_windows(self.W, self.S, self.FILTER))
            #print my_windows.shape
            if len(windows) == 0:
                windows = my_windows
            else: 
                windows = np.concatenate((windows,my_windows), axis=0)
        self.Kmeans_model = KMeans(n_clusters=K)
        self.Kmeans_model.fit(windows)
        #print windows.shape, 'win shape'
        self.cluster_centers = self.Kmeans_model.cluster_centers_
        return self.transform(sdf_list) #used to map back for sdf_class purposes/LSH pipeline

    def transform(self, sdf_list):
        features = np.array([])
        for file_ in sdf_list:
            #print file_
            converted = SDF(file_)
            my_windows = np.array(converted.make_windows(self.W, self.S, self.FILTER))
            prediction = histogramize(self.Kmeans_model.predict(my_windows), self.NUM_CLUSTERS)
            if len(features) == 0:
                features = prediction
            else:
                features = np.concatenate((features,prediction), axis=0)
        return features
        
