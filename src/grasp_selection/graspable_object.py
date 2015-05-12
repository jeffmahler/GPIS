'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
from abc import ABCMeta, abstractmethod

import numpy as np
import os
import sys

import mesh as m
import sdf as s
import similarity_tf as stf
import tfx

class GraspableObject():
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        if not isinstance(tf, stf.SimilarityTransform3D):
            raise ValueError('Must initialize graspable objects with 3D similarity transform')

        self.sdf_ = sdf
        self.mesh_ = mesh
        self.tf_ = tf
        self.grasps_ = {} # dictionary of grasps indexed by computation method?

        # make consistent poses, scales
        self.sdf_.tf = self.tf_
        if self.mesh_ is not None:
            self.mesh_.tf = self.tf_

    def sample_poses(self, num_samples, sigma_t = 0, sigma_r = 0):
        """ Samples pose perturbations of the shapes """
        todo = 1

    def sample_shapes(self, num_samples):
        """ Samples shape perturbations """
        todo = 1

    @property
    def sdf(self):
        return self.sdf_

    @property
    def mesh(self):
        return self.mesh_

    @property
    def tf(self):
        return self.tf_

    @property
    def pose(self):
        return self.tf_.pose

    @property
    def scale(self):
        return self.tf_.scale

    @tf.setter
    def tf(self, tf):
        """ Update the pose of the object wrt the world """
        self.tf_ = tf
        self.sdf.tf_ = tf
        if self.mesh_ is not None:
            self.mesh_.tf_ = tf

    @pose.setter
    def pose(self, pose):
        """ Update the pose of the object wrt the world """
        self.tf_.pose = pose
        self.sdf.tf_.pose = pose
        if self.mesh_ is not None:
            self.mesh_.tf_.pose = pose

    @scale.setter
    def scale(self, scale):
        """ Update the scale of the object wrt the world """
        self.tf_.scale = scale
        self.sdf.tf_.scale = scale
        if self.mesh_ is not None:
            self.mesh_.tf_.scale = scale

class GraspableObject2D(GraspableObject):
    # TODO: fix 2d with similiarity tfs
    def __init__(self, sdf, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf2D):
            raise ValueError('Must initialize graspable object 2D with 2D sdf')
        GraspableObject.__init__(self, sdf, tf=tf)

class GraspableObject3D(GraspableObject):
    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        if mesh is not None and not isinstance(mesh, m.Mesh3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        GraspableObject.__init__(self, sdf, mesh=mesh, tf=tf)
                
