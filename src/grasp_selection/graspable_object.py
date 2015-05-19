'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
from abc import ABCMeta, abstractmethod

import copy
import logging
import mayavi.mlab as mv
import numpy as np
import os
import sys

import mesh as m
import sdf as s
import similarity_tf as stf
import tfx

import IPython

class GraspableObject():
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', model_name='', category=''):
        if not isinstance(tf, stf.SimilarityTransform3D):
            raise ValueError('Must initialize graspable objects with 3D similarity transform')
        self.sdf_ = sdf
        self.mesh_ = mesh
        self.tf_ = tf

        self.key_ = key
        self.model_name_ = model_name # for OpenRave usage, gross!
        self.category_ = category

        # make consistent poses, scales
        self.sdf_.tf = self.tf_
        if self.mesh_ is not None:
            self.mesh_.tf = self.tf_

    @abstractmethod
    def transform(self, tf):
        """ Trasnsforms object by tf """
        pass

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

    @property
    def key(self):
        return self.key_

    @property
    def model_name(self):
        return self.model_name_

    @property
    def category(self):
        return self.category_

class GraspableObject2D(GraspableObject):
    # TODO: fix 2d with similiarity tfs
    def __init__(self, sdf, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf2D):
            raise ValueError('Must initialize graspable object 2D with 2D sdf')
        GraspableObject.__init__(self, sdf, tf=tf)

class GraspableObject3D(GraspableObject):
    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', category='',
                 model_name=''):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        if mesh is not None and not isinstance(mesh, m.Mesh3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')

        self.center_of_mass_ = sdf.center_world() # use SDF bb center for now
        GraspableObject.__init__(self, sdf, mesh=mesh, tf=tf, key=key, category=category, model_name=model_name)

    def visualize(self, com_scale = 0.01):
        """
        Display both mesh and center of mass at the given scale
        """
        if self.mesh_ is not None:
            self.mesh_.visualize()
            mv.points3d(self.center_of_mass_[0], self.center_of_mass_[1], self.center_of_mass_[2],
                        scale_factor=com_scale)

    def moment_arm(self, x):
        """ Computes the moment arm to point x """
        return x - self.center_of_mass_

    def transform(self, tf):
        """ Transforms object by tf """
        new_tf = tf.compose(self.tf_)
        sdf_tf = self.sdf_.transform(tf)

        # TODO: fix mesh class
        if self.mesh_ is not None:
            mesh_tf = copy.copy(self.mesh_)
            mesh_tf.tf_ = new_tf

        return GraspableObject3D(sdf_tf, mesh_tf, new_tf)

    def contact_friction_cone(self, contact, num_cone_faces = 4, friction_coef = 0.5):
        """
        Computes the friction cone and normal for a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj coords
            num_cone_faces - int number of cone faces to use
        Returns:
            success - False when cone can't be computed
            cone_support - numpy array where each column is a vector on the cone
            normal - direction vector
        """
        contact_grid = self.sdf.transform_pt_obj_to_grid(contact)
        on_surf, sdf_val = self.sdf.on_surface(contact_grid)
        if not on_surf:
            logging.debug('Contact point not on surface')
            return False, None, None

        grad = self.sdf.gradient(contact_grid)
        if np.all(grad == 0):
            return False, None, None

        # transform normal to obj frame
        normal = grad / np.linalg.norm(grad)
        normal = self.sdf.transform_pt_grid_to_obj(normal, direction = True)
        normal = normal.reshape((3, 1)) # make 2D for SVD

        # get tangent plane
        U, _, _ = np.linalg.svd(normal)

        # U[:, 1:] spans the tanget plane at the contact
        t1, t2 = U[:, 1], U[:, 2]
        tan_len = friction_coef
        force = -np.squeeze(normal) # shape of (3,) rather than (3, 1)
        cone_support = np.zeros((3, num_cone_faces))

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec
            
        return True, cone_support, normal

    def contact_torques(self, contact, forces):
        """
        Get the torques that can be applied by a set of vectors with a given friction cone
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            forces - numpt 3xN array of the forces applied at the contact
        Returns:
            success - bool, whether or not successful
            torques - numpy 3xN array of the torques that can be computed
        """
        contact_grid = self.sdf.transform_pt_obj_to_grid(contact)
        if not self.sdf.on_surface(contact_grid):
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self.moment_arm(contact)
        for i in range(num_forces):
            torques[:,i] = np.cross(moment_arm, forces[:,i])

        return True, torques
        

