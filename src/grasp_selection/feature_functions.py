from abc import ABCMeta, abstractmethod

import IPython
import json_serialization as jsons
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as ss

from contacts import SurfaceWindow
import graspable_object as go

class Weight:
    def __init__(self, value):
        self.value = value
        if isinstance(value, (float, int)):
            self.is_value = True
        elif isinstance(value, np.ndarray) and len(value.shape) == 1:
            self.is_value = True
        elif isinstance(value, str):
            self.is_value = False
        else:
            raise ValueError('Invalid weight: %s' %(value))

    def _window_width(self, feature):
        if len(feature.shape) == 1:
            num_elements = feature.shape[0]
            num_steps = int(np.sqrt(num_elements))
            if num_steps**2 == num_elements:
                return num_steps
        elif len(feature.shape) == 2 and feature.shape[0] == feature.shape[1]:
            return feature.shape[0]
        raise ValueError('Non-square feature vector')

    def extract_weight(self, key, other):
        if key.startswith('gaussian'): # (e.g. gaussian_3.0_0.1)
            _, max, sigma = key.split('_') # gaussian_MAX_SIGMA
            max, sigma = float(max), float(sigma)

            num_steps = self._window_width(other)
            gaussian_1d = ss.gaussian(num_steps, sigma)
            gaussian_2d = max * np.outer(gaussian_1d, gaussian_1d)
            return gaussian_2d.reshape(other.shape)
        elif key.startswith('crop'): # (e.g. crop_60.0_3)
            _, max, width = key.split('_') # crop_MAX_WIDTH
            max, width = float(max), int(width)

            center = max * np.ones((width, width))
            num_steps = self._window_width(other)
            offset = (num_steps - width) / 2
            crop_filter = np.pad(center, offset, 'constant', constant_values=0)
            return crop_filter.reshape(other.shape)
        else:
            raise ValueError('Invalid weight identifier: %s' %(key))

    def __mul__(self, other):
        if self.is_value:
            return self.value * other
        else:
            weight = self.extract_weight(self.value, other)
            return weight * other

    def __repr__(self):
        return repr(self.value)

# Grasp-specific feature functions

class FeatureExtractor:
    """Abstract class for extracting grasp features. The `phi` property method
    returns a feature vector."""
    __metaclass__ = ABCMeta
    use_unity_weights = False
    name = ''

    def __init__(self, feature_weight):
        self.feature_weight_ = feature_weight

    @property
    def feature_weight(self):
        return Weight(1.0) if self.use_unity_weights else self.feature_weight_

    def to_json(self, dest):
        return {}

    @property
    def phi(self):
        raise NotImplementedError

class AggregatedFeatureExtractor(FeatureExtractor):
    """Class for aggregating features."""
    def __init__(self, extractors, name):
        self.extractors_ = extractors
        self.name = name

    def to_json(self, dest):
        subdest = os.path.join(dest, self.name)
        try:
            os.makedirs(subdest)
        except os.error:
            pass

        subjson = {}
        for e in self.extractors_:
            subjson.update(e.to_json(subdest))
        return {self.name : subjson}

    @property
    def phi(self):
        phis = [e.phi for e in self.extractors_]
        if len(phis) == 0:
            return np.zeros(0)
        return np.concatenate(phis)

class GraspCenterFeatureExtractor(FeatureExtractor):
    name = 'center'

    """Class for extracting grasp center and direction."""
    def __init__(self, center, feature_weight=1.0):
        self.center_ = center
        self.feature_weight_ = feature_weight

    def to_json(self, dest):
        return {self.name: self.center_}

    @property
    def phi(self):
        return self.feature_weight_ * self.center_

class GraspAxisFeatureExtractor(FeatureExtractor):
    name = 'axis'

    """Class for extracting grasp center and direction."""
    def __init__(self, axis, feature_weight=1.0):
        self.axis_ = axis
        self.feature_weight_ = feature_weight

    def to_json(self, dest):
        return {self.name: self.axis_}

    @property
    def phi(self):
        return self.feature_weight_ * self.axis_

class GraspAxisAngleFeatureExtractor(FeatureExtractor):
    name = 'normal'

    """Class for extracting grasp center and direction."""
    def __init__(self, axis, normal, feature_weight=1.0):
        self.angle_ = self.angle(axis, normal)
        self.feature_weight_ = feature_weight

    def to_json(self, dest):
        return {self.name: self.axis_}

    def angle(self, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.arccos(np.dot(v1, v2))

    @property
    def phi(self):
        return self.feature_weight_ * np.array([self.angle_])

class WindowFeatureExtractor(FeatureExtractor):
    """Abstract class for extracting features from a grasp surface."""
    def __init__(self, surface, feature_weight=1.0):
        self.surface_ = surface
        self.feature_weight_ = feature_weight

    def to_json(self, dest):
        # create dest/self.name directory
        fname = os.path.join(dest, self.name)
        arr = self.phi.reshape((1, -1))
        np.savetxt(fname, arr, delimiter=',')
        # need to strip leading output_dir! currently in grasp_features.py
        return {self.name : fname}

class ProjectionWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting window features."""
    name = 'projection_window'

    @property
    def phi(self):
        return self.feature_weight * self.surface_.proj_win

class GradXWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting gradient wrt x features."""
    name = 'gradx_window'

    @property
    def phi(self):
        return self.feature_weight * self.surface_.grad_x

class GradYWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting gradient wrt y features."""
    name = 'grady_window'

    @property
    def phi(self):
        return self.feature_weight * self.surface_.grad_y

class CurvatureWindowFeatureExtractor(WindowFeatureExtractor):
    """Class for extracting curvature features."""
    name = 'curvature_window'

    @property
    def phi(self):
        return self.feature_weight * self.surface_.curvature

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

    def to_json(self, dest):
        return {self.name : self.phi}

class MomentArmFeatureExtractor(GravityFeatureExtractor):
    name = 'moment_arms'

    @property
    def phi(self):
        return self.feature_weight * np.r_[self.moment1_, self.moment2_]

class GraspAxisGravityAngleFeatureExtractor(GravityFeatureExtractor):
    name = 'grasp_axis_gravity_angle'

    @property
    def phi(self):
        angle = self.angle(self.grasp_.axis, self.gravity_force_)
        normalized_angle = np.pi - angle # flipped grasp axis should be same
        return self.feature_weight * np.array([normalized_angle])

class MomentArmGravityAngleFeatureExtractor(GravityFeatureExtractor):
    name = 'moment_arms_gravity_angle'

    @property
    def phi(self):
        angles = [self.angle(m, self.gravity_force_) for m in (self.moment1_, self.moment2_)]
        return self.feature_weight * np.array(angles)

GRAVITY_FORCE = 1.0 * np.array([0, 0, -9.8]) # TODO change this

# Graspable feature functions

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
        self.proj_win_weight_ = Weight(config['weight_proj_win'])
        self.grad_x_weight_ = Weight(config['weight_grad_x'])
        self.grad_y_weight_ = Weight(config['weight_grad_y'])
        self.curvature_weight_ = Weight(config['weight_curvature'])

        self.grasp_center_weight_ = Weight(config['weight_grasp_center'])
        self.grasp_axis_weight_ = Weight(config['weight_grasp_axis'])
        self.grasp_angle_weight_ = Weight(config['weight_grasp_angle'])

        self.gravity_weight_ = Weight(config['weight_gravity'])

        # for convenience
        self.weights_ = [
            self.proj_win_weight_, self.grad_x_weight_,
            self.grad_y_weight_, self.curvature_weight_
        ]
        self.classes_ = [
            ProjectionWindowFeatureExtractor, GradXWindowFeatureExtractor,
            GradYWindowFeatureExtractor, CurvatureWindowFeatureExtractor
        ]

    def _compute_feature_rep(self, grasp, root_name=None):
        """Extracts features from a graspable object and a single grasp."""
        # look in cache for features
        if grasp in self.features_:
            return self.features_[grasp]

        # get grasp windows
        try:
            s1, s2, c1, c2 = grasp.surface_information(self.graspable_,
                                                       self.window_width_, self.window_steps_)
        except ValueError as e:
            logging.warning('Failed to extract surface info with error');
            logging.warning(str(e))
            s1 = None
            s2 = None
            c1 = None
            c2 = None
            
        # if computing either surface fails, don't set surface_features
        if s1 is None or s2 is None or c1 is None or c2 is None:
            return None

        # compute surface features
        surface_features = []
        for s in (s1, s2):
            extractors = [cls(s, weight) for cls, weight in
                          zip(self.classes_, self.weights_)]
            feature = AggregatedFeatureExtractor(
                extractors, 'w%d' %(len(surface_features)+1))
            surface_features.append(feature)

        # compute grasp features
        grasp_pose_features = [
            GraspCenterFeatureExtractor(grasp.center, self.grasp_center_weight_),
            GraspAxisFeatureExtractor(grasp.axis, self.grasp_axis_weight_),
            # GraspAxisAngleFeatureExtractor(grasp.axis, c1.normal, self.grasp_angle_weight_),
            # GraspAxisAngleFeatureExtractor(-grasp.axis, c2.normal, self.grasp_angle_weight_)
        ]

        # compute gravity features
        gravity_args = (self.graspable_, grasp, GRAVITY_FORCE)
        gravity_features = [
            MomentArmFeatureExtractor(*gravity_args, feature_weight=self.gravity_weight_),
            GraspAxisGravityAngleFeatureExtractor(*gravity_args, feature_weight=0.0),
            MomentArmGravityAngleFeatureExtractor(*gravity_args, feature_weight=0.0),
        ]

        # compute additional features
        if root_name is None:
            root_name = self.graspable_.key
        features = AggregatedFeatureExtractor(
            surface_features + grasp_pose_features + gravity_features, root_name)
        self.features_[grasp] = features

        return features

    def compute_all_features(self, grasps):
        """Convenience function for extracting features from many grasps."""
        num_digits = len(str(len(grasps)-1)) # for padding with zeros
        features = []
        for i, grasp in enumerate(grasps):
            logging.info('Computing features for grasp %d' %(i))

            feature = self._compute_feature_rep(
                grasp, '%s_%s' %(self.graspable_.key, str(i).zfill(num_digits)))
            features.append(feature)
        return features

class GraspableFeatureLoader:
    """Class for loading pre-computed features for a graspable object."""
    def __init__(self, graspable, dataset, config):
        self.graspable_ = graspable
        self._parse_config(config)

        self.dataset_ = dataset
        self.features_dir_ = os.path.join(self.database_root_dir_,
                                          self.dataset_, 'features')

    def _parse_config(self, config):
        self.database_root_dir_ = config['database_dir']
        self.proj_win_weight_ = Weight(config['weight_proj_win'])
        self.grad_x_weight_ = Weight(config['weight_grad_x'])
        self.grad_y_weight_ = Weight(config['weight_grad_y'])
        self.curvature_weight_ = Weight(config['weight_curvature'])
        self.grasp_center_weight_ = Weight(config['weight_grasp_center'])
        self.grasp_axis_weight_ = Weight(config['weight_grasp_axis'])
        self.grasp_angle_weight_ = Weight(config['weight_grasp_angle'])
        self.gravity_weight_ = Weight(config['weight_gravity'])

    def _load_csv_as_array(self, rel_path):
        # rel_path is path to file relative to database root dir
        path = os.path.join(self.database_root_dir_, rel_path)
        return np.loadtxt(path, delimiter=',')

    def _load_feature_rep(self, data, grasp, root_name=None):
        name, feature_data = data.items()[0]
        surface_features = []
        for window in ('w1', 'w2'):
            # read window data from json
            surface_window_data = feature_data[window]
            curvature = self._load_csv_as_array(surface_window_data[CurvatureWindowFeatureExtractor.name])
            gradx = self._load_csv_as_array(surface_window_data[GradXWindowFeatureExtractor.name])
            grady = self._load_csv_as_array(surface_window_data[GradYWindowFeatureExtractor.name])
            proj_win = self._load_csv_as_array(surface_window_data[ProjectionWindowFeatureExtractor.name])
            surface_window = SurfaceWindow(proj_win, (gradx, grady), None, None, curvature)

            # create window feature extractors
            surface_extractors = [
                ProjectionWindowFeatureExtractor(surface_window, self.proj_win_weight_),
                GradXWindowFeatureExtractor(surface_window, self.grad_x_weight_),
                GradYWindowFeatureExtractor(surface_window, self.grad_y_weight_),
                CurvatureWindowFeatureExtractor(surface_window, self.curvature_weight_),
            ]
            extractor = AggregatedFeatureExtractor(surface_extractors, window)
            surface_features.append(extractor)

        # grasp features
        grasp_center = feature_data[GraspCenterFeatureExtractor.name]
        grasp_axis = feature_data[GraspAxisFeatureExtractor.name]
        grasp_pose_features = [
            GraspCenterFeatureExtractor(grasp_center, self.grasp_center_weight_),
            GraspAxisFeatureExtractor(grasp_axis, self.grasp_axis_weight_),
            # GraspAxisAngleFeatureExtractor(grasp.axis, c1.normal, self.grasp_angle_weight_),
            # GraspAxisAngleFeatureExtractor(-grasp.axis, c2.normal, self.grasp_angle_weight_)
        ]

        gravity_args = (self.graspable_, grasp, GRAVITY_FORCE)
        gravity_features = [
            MomentArmFeatureExtractor(*gravity_args, feature_weight=self.gravity_weight_),
            GraspAxisGravityAngleFeatureExtractor(*gravity_args, feature_weight=0.0),
            MomentArmGravityAngleFeatureExtractor(*gravity_args, feature_weight=0.0),
        ]

        # compute additional features
        if root_name is None:
            root_name = self.graspable_.key
        features = AggregatedFeatureExtractor(
            surface_features + grasp_pose_features + gravity_features, root_name)
        return features

    def load_all_features(self, grasps):
        path = os.path.join(self.features_dir_,
                            self.graspable_.key + '.json')
        with open(path) as f:
            features_json = jsons.load(f)
        num_digits = len(str(len(grasps)-1)) # for padding with zeros
        features = []
        for i, (feature_json, grasp) in enumerate(zip(features_json, grasps)):
            logging.info('Loading features for grasp %d' %(i))
            feature = self._load_feature_rep(
                feature_json, grasp, '%s_%s' %(self.graspable_.key, str(i).zfill(num_digits)))
            features.append(feature)
        return features
