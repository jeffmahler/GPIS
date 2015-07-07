from abc import ABCMeta, abstractmethod

import IPython
import logging
import numpy as np
import os

class FeatureFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, obj):
        """Evaluates a feature function with an object."""

class LinearFeatureFunction(FeatureFunction):
    def __init__(self, weights, dim=None):
        if dim is None:
            weights = np.array(weights)
        else:
            weights = weight * np.ones(dim)
        if len(weights.shape) > 2:
            raise ValueError('weights must have shape (1, N), (N, 1), or (N,)')
        if len(weights.shape) == 2 and 1 not in weights:
            raise ValueError('weights must have shape (1, N), (N, 1), or (N,)')
        self.weights_ = weights.reshape((max(weights.shape), 1))

    def evaluate(self, obj):
        obj = obj.reshape((max(obj.shape), 1))
        return np.dot(self.weights_.T, obj)

class LinearFeatureFunctionSum(LinearFeatureFunction):
    def __init__(self, feature_fns):
        self.feature_fns_ = feature_fns
        all_weights = [fn.weights_ for fn in feature_fns]
        self.weights_ = np.sum(np.array(all_weights), axis=0)
