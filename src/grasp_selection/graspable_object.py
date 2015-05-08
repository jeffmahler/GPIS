'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
import numpy as np
import os
import sys

import mesh as m
import sdf as s
import tfx

class GraspableObject():
    def __init__(self, sdf, mesh = None, pose = tfx.identity_tf(), scale = 1.0):
        if not isinstance(pose, tfx.canonical.CanonicalTransform):
            raise ValueError('Must initialize graspable objects with tfx canonical transform')

        self.sdf_ = sdf
        self.mesh_ = mesh
        self.pose_ = pose
        self.scale_ = scale
        self.grasps_ = {} # dictionary of grasps indexed by computation method?

        # make consistent poses, scales
        self.sdf_.pose = self.pose_
        if self.mesh_ is not None:
            self.mesh_.pose = self.pose_

        self.sdf_.scale = self.scale_
        if self.mesh_ is not None:
            self.mesh_.scale = self.scale_

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
    def pose(self):
        return self.pose_

    @property
    def scale(self):
        return self.scale_

    def set_pose(self, pose):
        """ Update the pose of the object wrt the world """
        self.pose_ = pose
        self.sdf_.pose = pose
        if self.mesh_ is not None:
            self.mesh_.pose = pose

    def set_scale(self, scale):
        """ Update the scale of the object wrt the world """
        self.scale_ = scale
        self.sdf_.scale = scale
        if self.mesh_ is not None:
            self.mesh_.scale = scale

class GraspableObject2D(GraspableObject):
    def __init__(self, sdf, pose = tfx.identity_tf(), scale = 1.0):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf2D):
            raise ValueError('Must initialize graspable object 2D with 2D sdf')
        GraspableObject.__init__(self, sdf, pose=pose, scale = scale)

class GraspableObject3D(GraspableObject):
    def __init__(self, sdf, mesh = None, pose = tfx.identity_tf(), scale = 1.0):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        if mesh is not None and not isinstance(mesh, m.Mesh3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        GraspableObject.__init__(self, sdf, mesh=mesh, pose=pose, scale = scale)
        
