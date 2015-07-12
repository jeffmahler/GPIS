"""
Contact class that encapsulates friction cone and surface window computation.
Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod
import numpy as np

class Contact:
    __metaclass__ = ABCMeta

class Contact3D(Contact):
    def __init__(self, graspable, contact_point):
        self.graspable_ = graspable
        self.point_ = contact_point # in world coordinates

        # cached attributes
        self.friction_cone_ = None
        self.normal_ = None # outward facing normal

    @property
    def graspable(self):
        return self.graspable_

    @property
    def point(self):
        return self.point_

    def friction_cone(self, num_cone_faces=4, friction_coef=0.5):
        """
        Computes the friction cone and normal for a contact point.
        Params:
            num_cone_faces - int number of cone faces to use
            friction_coef - float friction coefficient
        Returns:
            success - False when cone can't be computed
            cone_support - numpy array where each column is a vector on the cone
            normal - outward facing direction vector
        """
        if self.friction_cone_ is not None and self.normal_ is not None:
            return True, self.friction_cone_, self.normal_

        in_normal, t1, t2 = self.graspable._contact_tangents(self.point)
        if in_normal is None:
            return False, self.friction_cone_, self.normal_

        tan_len = friction_coef
        force = in_normal
        cone_support = np.zeros((3, num_cone_faces))

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec

        self.friction_cone_ = cone_support
        self.normal_ = -in_normal
        return True, self.friction_cone_, self.normal_

    def torques(self, forces):
        """
        Get the torques that can be applied by a set of vectors with a given
        friction cone.
        Params:
            forces - numpy 3xN array of the forces applied at the contact
        Returns:
            success - bool, whether or not successful
            torques - numpy 3xN array of the torques that can be computed
        """
        as_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        if not self.graspable.sdf.on_surface(as_grid):
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self.graspable.moment_arm(self.point)
        for i in range(num_forces):
            torques[:,i] = np.cross(moment_arm, forces[:,i])

        return True, torques

class SurfaceWindow:
    """Struct for encapsulating local surface window features."""
    def __init__(self, proj_win, grad, hess_x, hess_y, gauss_curvature):
        self.proj_win_ = proj_win
        self.grad_ = grad
        self.hess_x_ = hess_x
        self.hess_y_ = hess_y
        self.gauss_curvature_ = gauss_curvature

    @property
    def proj_win(self):
        return self.proj_win_.flatten()

    @property
    def grad_x(self):
        return self.grad_[0].flatten()

    @property
    def grad_y(self):
        return self.grad_[1].flatten()

    @property
    def curvature(self):
        return self.gauss_curvature_.flatten()

    def asarray(self, proj_win_weight=0.0, grad_x_weight=0.0,
                grad_y_weight=0.0, curvature_weight=0.0):
        proj_win = proj_win_weight * self.proj_win
        grad_x = grad_x_weight * self.grad_x
        grad_y = grad_y_weight * self.grad_y
        curvature = curvature_weight * self.gauss_curvature
        return np.append([], [proj_win, grad_x, grad_y, curvature])
