"""
Gaussian Process implicit surfaces for shape uncertainty
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import GPy as gpy
import sdf
import logging

class Gpis(Sdf):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def sample_shape(self, num_samples):
        """
        Creates a number of samples SDFs from the GPIS
        Params:
            num_samples: (int) number of shapes to sample
        Returns:
            list of Sdfs
        """
        pass

    @abstractmethod
    def predict_locations(self, points):
        """
        Predict a number of SDF locations with mean and variance
        Params:
            points: (nxd) numpy array of points
        Returns:
            means: (nx1) numpy array of means
            vars: (nx1) numpy array of Gaussian vars
        """
        pass

class Gpis2D(Gpis):
    def __init__(self, points, sdf_meas, meas_variance, kernel_name = 'rbf', kernel_hyps = [1., 1.],
                 origin = np.array([0,0]), resolution = 1.0, pose = tfx.identity_tf(frame="world")):
        self.points_ = points
        self.data_ = means
        self.vars = variances

        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape
        self.pose_ = pose

        # set up gp
        self.set_kernel(kernel_name, kernel_hyps)
        self.gp_ = gpy.models.GPRegression(points, sdf_meas, self.kernel_, Y_variance=meas_variance)

        self.feature_vector_ = None #Kmeans feature representation

    def set_kernel(self, kernel_name):
        if kernel_name == 'rbf':
            self.kernel_ = gpy.kern.RBF(input_dim=2, variance=kernel_hyps[0], lengthscale=kernel_hyps[1])
        else:
            raise ValueError('Invalid Kernel specified')

    def __getitem__(self, coords):
        """
        Returns the signed distance at the given coordinates, interpolating if necessary
        Params: numpy 2 array
        Returns:
            float: the signed distance and the given coors (interpolated)
        """
        if len(coords) != 2:
            raise IndexError('Indexing must be 2 dimensional')

        # log warning if out of bounds access
        if coords[0] < 0 or coords[0] >= self.dims_[0] or coords[0] < 0 or coords[1] >= self.dims_[1]:
            logging.warning('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        new_coords = np.zeros([2,])
        new_coords[0] = max(0, min(coords[0], self.dims_[0]))
        new_coords[1] = max(0, min(coords[1], self.dims_[1]))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int:
            new_coords = new_coords.astype(np.int)
            return self.data_[new_coords[0], new_coords[1]]

        # otherwise interpolate
        min_coords = np.floor(new_coords)
        max_coords = np.ceil(new_coords)
        points = np.array([[min_coords[0], min_coords[1]],
                           [max_coords[0], min_coords[1]],
                           [min_coords[0], max_coords[1]],
                           [max_coords[0], max_coords[1]]])

        num_interpolants= 4
        values = np.zeros([num_interpolants,])
        weights = np.ones([num_interpolants,])
        for i in range(num_interpolants):
            p = points[i,:].astype(np.int)
            values[i] = self.data_[p[0], p[1]]
            dist = np.linalg.norm(new_coords - p.T)
            if dist > 0:
                weights[i] = 1.0 / dist

        if np.sum(weights) == 0:
            weights = np.ones([num_interpolants,])
        weights = weights / np.sum(weights)
        return weights.dot(values)

    def imshow(self):
        """
        Displays the SDF as an image
        """
        plt.figure()
        plt.imshow(self.mean_vec_)
        plt.show()
