import numpy as np
import os
import sys

import IPython

import features as f

HEADER_SIZE = 8 #size of feature file header

def string_to_array(string):
    """
    Takes in a string of space-separated values and returns an np-array of those values
    """
    return np.array([float(i) for i in string.split()])

class LocalFeatureFile:
    '''
    For reading and writing DexNet custom local feature desriptor files (.ftr)
    '''
    def __init__(self, filepath):
        '''
        Set the path to the file to open
        '''
        self.filepath_ = filepath
        file_root, file_ext = os.path.splitext(self.filepath_)
        if file_ext != '.ftr' and file_ext != '.txt':
            print 'Extension', file_ext
            raise Exception('Cannot load file extension %s. Please supply a .ftr file' %(file_ext))

    @property
    def filepath(self):
        '''
        Returns the path to the file to read / write
        '''
        return self.filepath_

    def read(self):
        '''
        Read in the feature file. Currently hardcorded for SHOT features, will change later
        '''
        feature_file = open(self.filepath_, 'r')
        num_descriptors = int(feature_file.readline())
        len_descriptors = int(feature_file.readline())
        len_rf = int(feature_file.readline())
       
        # initialize features object
        features = f.BagOfFeatures()

        # parse through the lines of data from the shot file to fill out the matrices
        for i in range(num_descriptors):
            p = feature_file.readline().split('\t')
            rf, descriptor, keypoint, normal = [string_to_array(j) for j in p] 
            if np.sum(np.isnan(descriptor)) == 0 and np.sum(np.isinf(descriptor)) == 0:
                features.add(f.LocalFeature(descriptor, rf, keypoint, normal))

        feature_file.close()

        # return feature object
        return features

    def write(self, mesh):
        '''
        Write a mesh to an obj file.
        Assumes mesh vertices, faces, and normals are in standard python list objects.
        Does not support material files or texture coordinates
        '''
        raise Exception('FeatureFile writes not yet supported')

if __name__ == '__main__':
    feature_file = 'data/features/5c74962846d6cd33920ed6df8d81211d_features.txt'
    a = LocalFeatureFile(feature_file)
    feat = a.read()
