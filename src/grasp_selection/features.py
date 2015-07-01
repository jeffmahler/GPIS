from abc import ABCMeta, abstractmethod

import numpy as np

class Feature:
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

class LocalFeature(Feature):
    __metaclass__ = ABCMeta

    def __init__(self, descriptor, rf, point, normal):
        self.descriptor_ = descriptor
        self.rf_ = rf
        self.point_ = point
        self.normal_ = normal

    @property
    def descriptor(self):
        return self.descriptor_

    @property
    def reference_frame(self):
        return self.rf_

    @property
    def keypoint(self):
        return self.point_

    @property
    def normal(self):
        return self.normal_

class SHOTFeature(LocalFeature):
    """ Same interface as standard local feature """ 
    def __init__(self, descriptor, rf, point, normal):
        LocalFeature.__init__(self, descriptor, rf, point, normal)

class BagOfFeatures:
    """ Actually just a list of features, but created for the sake of future bag-of-words reps """
    def __init__(self, features = None):
        self.features_ = features
        if self.features_ is None:
            self.features_ = []

        self.num_features_ = len(self.features_)

    def add(self, feature):
        """ Add a new feature to the bag """
        self.features_.append(feature)
        self.num_features_ = len(self.features_)        

    def extend(self, features):
        """ Add a list of features to the bag """
        self.features_.extend(features)
        self.num_features_ = len(self.features_)        

    def feature(self, index):
        """ Returns a feature """
        if index < 0 or index >= self.num_features_:
            raise ValueError('Index %d out of range' %(index))
        return self.features_[index]

    def feature_subset(self, indices):
        """ Returns some subset of the features """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            raise ValueError('Can only index with lists')
        return [self.features_[i] for i in indices]

    @property
    def num_features(self):
        return self.num_features_

    @property
    def descriptors(self):
        """ Make a nice array of the descriptors """
        return np.array([f.descriptor for f in self.features_])

    @property
    def reference_frames(self):
        """ Make a nice array of the reference frames """
        return np.array([f.reference_frame for f in self.features_])

    @property
    def keypoints(self):
        """ Make a nice array of the keypoints """
        return np.array([f.keypoint for f in self.features_])

    @property
    def normals(self):
        """ Make a nice array of the normals """
        return np.array([f.normal for f in self.features_])
