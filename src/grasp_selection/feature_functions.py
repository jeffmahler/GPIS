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

class FeatureExtractor:
    """Abstract class for extracting grasp features. The `phi` property method
    returns a feature vector."""
    __metaclass__ = ABCMeta
    use_unity_weights = False

    def __init__(self, feature_weight):
        self.feature_weight_ = feature_weight

    @property
    def feature_weight(self):
        return 1.0 if self.use_unity_weights else self.feature_weight_

    @property
    def phi(self):
        raise NotImplementedError

class WindowFeatureExtractor(FeatureExtractor):
    """Abstract class for extracting features from a grasp surface."""
    def __init__(self, surface, feature_weight=1.0):
        self.surface_ = surface
        self.feature_weight_ = feature_weight

class ProjectionWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting window features."""

    @property
    def phi(self):
        return self.feature_weight * self.surface_.proj_win

class GradXWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting gradient wrt x features."""

    @property
    def phi(self):
        return self.feature_weight * self.surface_.grad_x

class GradYWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting gradient wrt y features."""

    @property
    def phi(self):
        return self.feature_weight * self.surface_.grad_y

class CurvatureWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting curvature features."""

    @property
    def phi(self):
        return self.feature_weight * self.surface_.curvature

class SurfaceWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for concatenating window features."""
    def __init__(self, extractors):
        self.extractors_ = extractors

    @property
    def phi(self):
        phis = [e.phi for e in self.extractors_]
        return np.concatenate(phis)

class GravityFeatureExtractor(FeatureExtractor):
    """Abstract class for extracting gravity-related features."""
    def __init__(self, graspable, grasp, gravity_force, feature_weight=1.0):
        self.graspable_ = graspable
        self.grasp_ = grasp
        self.gravity_force_ = gravity_force # np 3 array, e.g. np.array([0, 0, -mg])
        self.feature_weight_ = feature_weight

        # Compute moment arms
        _, (c1, c2) = grasp.close_fingers(graspable)
        self.moment1_ = self.graspable_.moment_arm(c1.point)
        self.moment2_ = self.graspable_.moment_arm(c2.point)

    def angle(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.arccos(np.dot(v1, v2))

class MomentArmFeatureExtractor(GravityFeatureExtractor):
    @property
    def phi(self):
        return self.feature_weight * np.r_[self.moment1_, self.moment2_]

class GraspAxisGravityAngleFeatureExtractor(GravityFeatureExtractor):
    @property
    def phi(self):
        angle = self.angle(self.grasp_.axis, self.gravity_force_)
        normalized_angle = np.pi - angle # flipped grasp axis should be same
        return self.feature_weight * np.array([normalized_angle])

class MomentArmGravityAngleFeatureExtractor(GravityFeatureExtractor):
    @property
    def phi(self):
        angles = [self.angle(m, self.gravity_force_) for m in (self.moment1_, self.moment2_)]
        return self.feature_weight * np.array(angles)

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
            ProjectionWindowFeatureExtractor, GradXWindowFeatureExtractor,
            GradYWindowFeatureExtractor, CurvatureWindowFeatureExtractor
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
            feature = SurfaceWindowFeatureExtractor(extractors)
            surface_features.append(feature)

        # compute gravity features
        gravity_force = 1.0 * np.array([0, 0, -9.8])
        gravity_args = (self.graspable_, grasp, gravity_force)
        gravity_features = [
            MomentArmFeatureExtractor(*gravity_args),
            GraspAxisGravityAngleFeatureExtractor(*gravity_args),
            MomentArmGravityAngleFeatureExtractor(*gravity_args),
        ]

        # compute additional features
        features = surface_features + gravity_features
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
