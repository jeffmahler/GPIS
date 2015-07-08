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

# Grasp-specific feature functions

class SurfaceWindow:
    """Struct for encapsulating local surface window features."""
    def __init__(self, proj_win, grad, hess_x, hess_y, gauss_curvature):
        self.proj_win_ = proj_win
        self.grad_ = grad
        self.hess_x_ = hess_x
        self.hess_y_ = hess_y
        self.gauss_curvature_ = gauss_curvature

    @property
    def proj_win(self):
        return self.proj_win_.flatten()

    @property
    def grad_x(self):
        return self.grad_[0].flatten()

    @property
    def grad_y(self):
        return self.grad_[1].flatten()

    @property
    def curvature(self):
        return self.gauss_curvature_.flatten()

    def asarray(self, proj_win_weight=0.0, grad_x_weight=0.0,
                grad_y_weight=0.0, curvature_weight=0.0):
        proj_win = proj_win_weight * self.proj_win
        grad_x = grad_x_weight * self.grad_x
        grad_y = grad_y_weight * self.grad_y
        curvature = curvature_weight * self.gauss_curvature
        return np.append([], [proj_win, grad_x, grad_y, curvature])


class GraspFeatureExtractor:
    __metaclass__ = ABCMeta

    name = ''

    def __init__(self, surface, weight):
        self.surface_ = surface
        self.weight_ = weight

    @property
    def phi(self):
        raise NotImplementedError

class WindowGraspFeatureExtractor(GraspFeatureExtractor):
    name = 'proj_win'

    @property
    def phi(self):
        return self.weight_ * self.surface_.proj_win

class GradXGraspFeatureExtractor(GraspFeatureExtractor):
    name = 'grad_x'

    @property
    def phi(self):
        return self.weight_ * self.surface_.grad_x

class GradYGraspFeatureExtractor(GraspFeatureExtractor):
    name = 'grad_y'

    @property
    def phi(self):
        return self.weight_ * self.surface_.grad_y

class CurvatureGraspFeatureExtractor(GraspFeatureExtractor):
    name = 'curvature'

    @property
    def phi(self):
        return self.weight_ * self.surface_.curvature

class SurfaceGraspFeatureExtractor(GraspFeatureExtractor):
    def __init__(self, extractors):
        self.extractors_ = extractors
        for e in extractors:
            # for example, if one of the extractors is a
            # WindowGraspFeatureExtractor, set the self.proj_win
            # attribute to that extractor
            assert not hasattr(self, e.name), 'Extractors must be unique.'
            setattr(self, e.name, e)

    @property
    def phi(self):
        phis = [e.phi for e in self.extractors_]
        return np.concatenate(phis)
