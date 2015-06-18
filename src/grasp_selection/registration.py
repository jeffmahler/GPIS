from abc import ABCMeta, abstractmethod

import IPython
import numpy as np

class RegistrationSolver:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def register(self, correspondences):
        """ Register objects to one another """
        pass

class RigidRegistrationSolver(RegistrationSolver):
    def __init__(self):
        pass

    def register(self, correspondences, weights=None):
        """ Register objects to one another """
        # setup the problem
        source_points = correspondences.source_points
        target_points = correspondences.target_points
        N = correspondences.num_matches

        if weights is None:
            weights = np.ones([correspondences.num_matches, 1])
        if weights.shape[1] == 1:
            weights = np.tile(weights, (1, 3)) # tile to get to 3d space

        # calculate centroids (using weights)
        source_centroid = np.sum(weights * source_points, axis=0) / np.sum(weights, axis=0)
        target_centroid = np.sum(weights * target_points, axis=0) / np.sum(weights, axis=0)
        
        # center the datasets
        source_centered_points = source_points - np.tile(source_centroid, (N,1))
        target_centered_points = target_points - np.tile(target_centroid, (N,1))

        # find the covariance matrix and finding the SVD
        H = np.dot((weights * source_centered_points).T, weights * target_centered_points)
        U, S, V = np.linalg.svd(H) # this decomposes H = USV, so V is "V.T"

        # calculate the rotation
        R = np.dot(V.T, U.T)
        
        # special case (reflection)
        if np.linalg.det(R) < 0:
                V[2,:] *= -1
                R = np.dot(V.T, U.T)
        
        # calculate the translation + concatenate the rotation and translation
        t = np.matrix(np.dot(-R, source_centroid) + target_centroid)
        tf_source_target = np.hstack([R, t.T])
        return tf_source_target
