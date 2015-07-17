from abc import ABCMeta, abstractmethod

import IPython
import logging
import numpy as np
import os

# Unused FeatureFunction classes

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

class GraspFeatureExtractor:
    """Abstract class for extracting features from a grasp surface. The `phi`
    property method returns a feature vector."""
    __metaclass__ = ABCMeta

    def __init__(self, surface, weight):
        self.surface_ = surface
        self.weight_ = weight

    @property
    def phi(self):
        raise NotImplementedError

class WindowGraspFeatureExtractor(GraspFeatureExtractor):
    """Class for extracting window features."""

    @property
    def phi(self):
        return self.weight_ * self.surface_.proj_win

class GradXGraspFeatureExtractor(GraspFeatureExtractor):
    """Class for extracting gradient wrt x features."""

    @property
    def phi(self):
        return self.weight_ * self.surface_.grad_x

class GradYGraspFeatureExtractor(GraspFeatureExtractor):
    """Class for extracting gradient wrt y features."""

    @property
    def phi(self):
        return self.weight_ * self.surface_.grad_y

class CurvatureGraspFeatureExtractor(GraspFeatureExtractor):
    """Class for extracting curvature features."""

    @property
    def phi(self):
        return self.weight_ * self.surface_.curvature

class SurfaceGraspFeatureExtractor(GraspFeatureExtractor):
    """Class for concatenating features."""
    def __init__(self, extractors):
        self.extractors_ = extractors

    @property
    def phi(self):
        phis = [e.phi for e in self.extractors_]
        return np.concatenate(phis)

class GraspableFeatureExtractor:
    """Class for extracting features from a graspable object and an arbitrary
    number of grasps."""
    def __init__(self, graspable, config):
        self.graspable_ = graspable
        self.features_ = {} # to cache feature computation
        self._parse_config(config)

    def _parse_config(self, config):
        # featurization
        self.window_width_ = config['window_width']
        self.window_steps_ = config['window_steps']
        self.window_sigma_ = config['window_sigma']

        # feature weights
        self.proj_win_weight_ = config['weight_proj_win']
        self.grad_x_weight_ = config['weight_grad_x']
        self.grad_y_weight_ = config['weight_grad_y']
        self.curvature_weight_ = config['weight_curvature']

        # for convenience
        self.weights_ = [
            self.proj_win_weight_, self.grad_x_weight_,
            self.grad_y_weight_, self.curvature_weight_
        ]
        self.classes_ = [
            WindowGraspFeatureExtractor, GradXGraspFeatureExtractor,
            GradYGraspFeatureExtractor, CurvatureGraspFeatureExtractor
        ]

    def _compute_feature_rep(self, grasp):
        """Extracts features from a graspable object and a single grasp."""
        # get grasp windows -- cached
        try:
            s1, s2 = grasp.surface_information(self.graspable_,
                                               self.window_width_, self.window_steps_)
        except ValueError as e:
            logging.warning('Failed to extract surface info with error');
            logging.warning(str(e))
            s1 = None
            s2 = None

        # if computing either surface fails, don't set surface_features
        if s1 is None or s2 is None:
            return

        # look in cache for features
        if grasp in self.features_:
            return self.features_[grasp]

        # compute surface features
        surface_features = []
        for s in (s1, s2):
            extractors = [cls(s, weight) for cls, weight in
                          zip(self.classes_, self.weights_) if weight > 0]
            feature = SurfaceGraspFeatureExtractor(extractors)
            surface_features.append(feature)

        # compute additional features
        features = list(surface_features)
        # features.append(GravityGraspFeatureExtractor(...)) # TODO
        self.features_[grasp] = features

        return features

    def compute_all_features(self, grasps):
        """Convenience function for extracting features from many grasps."""
        features = []
        for i, grasp in enumerate(grasps):
            logging.info('Computing features for grasp %d' %(i))
            feature = self._compute_feature_rep(grasp)
            features.append(feature)
        return features
