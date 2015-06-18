from abc import ABCMeta, abstractmethod

import IPython
import os
import sys

import feature_file as ff
import graspable_object as go
import obj_file
import sdf_file

class FeatureExtractor:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def extract(self, graspable):
        """ Returns a set of extracted features for a graspable """
        pass

class SHOTFeatureExtractor(FeatureExtractor):
    key = 'shot'

    def __init__(self):
        pass

    def extract(self, graspable, feature_file_name):
        """ Returns a set of extracted SHOT features for a graspable """
        # make OS call to extract features to disk
        shot_os_call = 'bin/shot_extractor %s %s' %(graspable.model_name, feature_file_name)
        os.system(shot_os_call)

        # add features to graspable and return
        feature_file = ff.LocalFeatureFile(feature_file_name)
        shot_features = feature_file.read()
        graspable.features[SHOTFeatureExtractor.key] = shot_features

if __name__ == '__main__':
    # load object
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = go.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    shot_features_name = 'data/test/features/Co_clean_features.txt'
    
    feature_extractor = SHOTFeatureExtractor()
    feature_extractor.extract(graspable, shot_features_name)

    IPython.embed()
