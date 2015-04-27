'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
import numpy as np
import os
import sys

import mesh
import sdf

class GraspableObject():
    def __init__(self, sdf, mesh = None, pose = None):
        # TODO: load mesh, sdf
        
        self.grasps_ = []

        # TODO: figure out whether or not a good pose class already exists (e.g. tfx)
        todo = 1

    def sample_poses(self, num_samples, sigma_t = DEF_SIGMA_T, sigma_r = DEF_SIGMA_R):
        '''
        Samples pose perturbations of the shapes
        '''
        todo = 1

    def sample_shapes(self, num_samples):
        '''
        Samples shape perturbations
        '''
        todo = 1


class GraspableObject2D(GraspableObject):
    def sdf(self, x, y):
        '''
        Returns the distribution on signed distance value at the given coords
        '''
        return self.sdf_[x, y]

class GraspableObject3D(GraspableObject):
    def sdf(self, x, y, z):
        '''
        Returns the distribution on signed distance value at the given coords
        '''
        return self.sdf_[x, y, z]
        
