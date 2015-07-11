"""
Contact class that encapsulates friction cone and surface window computation.
Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod

class Contact:
    __metaclass__ = ABCMeta

class Contact3D(Contact):
    def __init__(self, graspable, contact_point):
        self.graspable_ = graspable
        self.point_ = contact_point # in world coordinates

    @property
    def graspable(self):
        return self.graspable_

    @property
    def point(self):
        return self.point_

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
