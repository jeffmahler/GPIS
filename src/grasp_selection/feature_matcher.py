from abc import ABCMeta, abstractmethod

import features as f
import numpy as np
import IPython
from scipy import spatial

class Correspondences:
    def __init__(self, index_map, source_points, target_points):
        self.index_map_ = index_map
        self.source_points_ = source_points
        self.target_points_ = target_points
        self.num_matches_ = source_points.shape[0]

    @property
    def index_map(self):
        return self.index_map_

    @property
    def source_points(self):
        return self.source_points_

    @property
    def target_points(self):
        return self.target_points_

    @property
    def num_matches(self):
        return self.num_matches_

    # Functions to iterate through matches like "for source_corr, target_corr in correspondences"
    def __iter__(self):
        self.iter_count_ = 0

    def next(self):
        if self.iter_count_ >= len(self.num_matches_):
            raise StopIteration
        else:
            return self.source_points_[self.iter_count,:], self.target_points_[self.iter_count,:]

class FeatureMatcher:
    """
    Generic feature matching between local features on a source and target object using nearest neighbors
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def get_point_index(point, all_points, eps = 1e-4):
        """ Get the index of a point in an array """
        inds = np.where(np.linalg.norm(point - all_points, axis=1) < eps)
        if inds[0].shape[0] == 0:
            return -1
        return inds[0][0]
    
    @abstractmethod
    def match(self, source_obj, target_obj):
        """
        Matches features between two graspable objects
        """
        pass

class RawDistanceFeatureMatcher(FeatureMatcher):
    def match(self, source_obj_features, target_obj_features):
        """
        Matches features between two graspable objects based on a full distance matrix
        Params:
            source_obj_features: (BagOfFeatures) bag of the source objects features
            target_obj_features: (BagOfFeatures) bag of the target objects features
        Returns:
            corrs: (Correspondences) the correspondences between source and target 
        """
        if not isinstance(source_obj_features, f.BagOfFeatures):
            raise ValueError('Must supply source bag of object features')
        if not isinstance(target_obj_features, f.BagOfFeatures):
            raise ValueError('Must supply target bag of object features')

        # source feature descriptors and keypoints
        source_descriptors = source_obj_features.descriptors
        target_descriptors = target_obj_features.descriptors
        source_keypoints = source_obj_features.keypoints
        target_keypoints = target_obj_features.keypoints

        #calculate distance between this model's descriptors and each of the other_model's descriptors
        dists = spatial.distance.cdist(source_descriptors, target_descriptors) 
        
        #calculate the indices of the target_model that minimize the distance to the descriptors in this model
        source_closest_descriptors = dists.argmin(axis=1)
        target_closest_descriptors = dists.argmin(axis=0)
        match_indices = []
        source_matched_points = np.zeros((0,3))
        target_matched_points = np.zeros((0,3))

        #calculate which points/indices the closest descriptors correspond to
        for i, j in enumerate(source_closest_descriptors):
            # for now, only keep correspondences that are a 2-way match
            if target_closest_descriptors[j] == i:
                match_indices.append(j)
                source_matched_points = np.r_[source_matched_points, source_keypoints[i:i+1, :]]
                target_matched_points = np.r_[target_matched_points, target_keypoints[j:j+1, :]]
            else:
                match_indices.append(-1)

        return Correspondences(match_indices, source_matched_points, target_matched_points)

