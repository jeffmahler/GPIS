"""
Gaussian Process implicit surfaces for shape uncertainty
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import GPy as gpy
import sdf
import sdf_file

from PIL import Image
import scipy.io
import scipy.ndimage
import scipy.signal
from skimage import feature
import skimage.filters

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tfx

import IPython

class Gpis(sdf.Sdf):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def sample_sdfs(self, num_samples):
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

class Gpis3D(Gpis):
    def __init__(self, points, sdf_meas, dims, meas_variance = 0.1, kernel_name = 'rbf', kernel_hyps = [1., 3.],
                 origin = np.array([0,0]), resolution = 1.0, pose = tfx.identity_tf(frame="world")):
        if len(dims) != 3:
            raise ValueError('Dimensions must be 3D!')
        
        # store params
        self.pts_ = points
        self.data_ = sdf_meas
        self.var_ = meas_variance
        self.dims_ = dims

        self.origin_ = origin
        self.resolution_ = resolution
        self.pose_ = pose

        self._compute_flat_indices()

        # set up gp
        self.set_kernel(kernel_name, kernel_hyps)
        self.gp_ = gpy.models.GPRegression(points, sdf_meas, self.kernel_, Y_variance=meas_variance)
        self.gp_.optimize()
        print(self.gp_)

        # predict mean and variance of grid
        (self.mean_pred_, self.var_pred_) = self.predict_locations(self.grid_pts_, full_cov=False)

        self.feature_vector_ = None #Kmeans feature representation

    def set_kernel(self, kernel_name, kernel_hyps):
        """
        Sets the kernel of the Gaussian Process
        """
        if kernel_name == 'rbf':
            self.kernel_ = gpy.kern.RBF(input_dim=2, variance=kernel_hyps[0], lengthscale=kernel_hyps[1])
        else:
            raise ValueError('Invalid Kernel specified')

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind, z_ind] = np.indices(self.dims_)
        self.grid_pts_ = np.c_[x_ind.flatten().T, np.c_[y_ind.flatten().T, z_ind.flatten().T]]

    def sample_sdfs(self, num_samples=1, full_cov=True):
        """
        Samples sdfs from the GPIS
        Params:
            num_samples: (int) number of samples to generate
            full_cov: (bool) whether or not to use the diagonal or entire covariance matrix
        Returns:
            list of sdf objects
        """        
        sdf_samples = self.gp_.posterior_samples(self.grid_pts_, num_samples)
        sdfs = []
        for i in range(num_samples):
            sdf_data = sdf_samples[:,i].reshape(self.dims_)
            sdfs.append(sdf.Sdf3D(sdf_data, pose = self.pose_))
        return sdfs

    def predict_locations(self, points, full_cov=False):
        """
        Predicts the sdf values at the given points
        Params:
            points: (num_points x 2 numpy array) points to predict
            full_cov: (bool) whether or not to use the diagonal or entire covariance matrix
        Returns:
            numpy array (num_pts x 1) containing predictions
        """        
        return self.gp_.predict(points, full_cov=full_cov)

    def surface_points(self, surface_thresh = sdf.DEF_SURFACE_THRESH):
        """
        Returns the points on the surface of the mean sdf
        Params: (float) sdf value to threshold
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        return self.mean_sdf().surface_points()

    def __getitem__(self, coords):
        """
        Returns the signed distance at the given coordinates, interpolating if necessary
        Params: numpy 3 array
        Returns:
            float: the signed distance and the given coors (interpolated)
        """
        if len(coords) != 3:
            raise IndexError('Indexing must be 3 dimensional')

        # log warning if out of bounds access
        if coords[0] < 0 or coords[0] >= self.dims_[0] or coords[0] < 0 or coords[1] >= self.dims_[1] or coords[2] < 0 or coords[2] >= self.dims_[2]:
            logging.warning('Out of bounds access. Snapping to SDF dims')

        # snap to grid dims
        new_coords = np.zeros([3,])
        new_coords[0] = max(0, min(coords[0], self.dims_[0]))
        new_coords[1] = max(0, min(coords[1], self.dims_[1]))
        new_coords[2] = max(0, min(coords[2], self.dims_[2]))

        # regular indexing if integers
        if type(coords[0]) is int and type(coords[1]) is int and type(coords[2]) is int:
            new_coords = new_coords.astype(np.int)
            return self.data_[new_coords[0], new_coords[1], new_coords[2]]

        # otherwise interpolate
        min_coords = np.floor(new_coords)
        max_coords = np.ceil(new_coords)
        points = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                           [max_coords[0], min_coords[1], min_coords[2]],
                           [min_coords[0], max_coords[1], min_coords[2]],
                           [min_coords[0], min_coords[1], max_coords[2]],
                           [max_coords[0], max_coords[1], min_coords[2]],
                           [min_coords[0], max_coords[1], max_coords[2]],
                           [max_coords[0], min_coords[1], max_coords[2]],
                           [max_coords[0], max_coords[1], max_coords[2]]])

        num_interpolants = 8
        values = np.zeros([num_interpolants,])
        weights = np.ones([num_interpolants,])
        for i in range(num_interpolants):
            p = points[i,:].astype(np.int)
            values[i] = self.data_[p[0], p[1], p[2]]
            dist = np.linalg.norm(new_coords - p.T)
            if dist > 0:
                weights[i] = 1.0 / dist

        if np.sum(weights) == 0:
            weights = np.ones([num_interpolants,])
        weights = weights / np.sum(weights)
        return weights.dot(values)

    def scatter(self, surface_thresh = sdf.DEF_SURFACE_THRESH):
        """
        Plots the SDF as a matplotlib 3D scatter plot, and displays the figure
        Params: - 
        Returns: - 
        """
        h = plt.figure()
        ax = h.add_subplot(111, projection = '3d')

        surface_points = np.where(np.abs(self.mean_pred_.reshape(self.dims_)) < surface_thresh)

        # get the points on the surface
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]

        # scatter the surface points
        ax.scatter(x, y, z, cmap="Blues")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(0, self.dims_[0])
        ax.set_ylim3d(0, self.dims_[1])
        ax.set_zlim3d(0, self.dims_[2])
        plt.show()


class Gpis2D(Gpis):
    def __init__(self, points, sdf_meas, dims, meas_variance = 0.1, kernel_name = 'rbf', kernel_hyps = [100., 5.0],
                 origin = np.array([0,0]), resolution = 1.0, pose = tfx.identity_tf(frame="world")):
        if len(dims) != 2:
            raise ValueError('Dimensions must be 3D!')

        # store params
        self.pts_ = points
        self.data_ = sdf_meas
        self.var_ = meas_variance
        self.dims_ = dims

        self.origin_ = origin
        self.resolution_ = resolution
        self.pose_ = pose

        self._compute_flat_indices()

        # set up gp
        self.set_kernel(kernel_name, kernel_hyps)
        self.gp_ = gpy.models.GPRegression(points, sdf_meas, self.kernel_, Y_variance=meas_variance)
        #        self.gp_.optimize(max_f_eval=1)
        print(self.gp_)

        # predict mean and variance of grid
        (self.mean_pred_, self.var_pred_) = self.predict_locations(self.grid_pts_, full_cov=True)

        self.feature_vector_ = None #Kmeans feature representation

    def set_kernel(self, kernel_name, kernel_hyps):
        """
        Sets the kernel of the Gaussian Process
        """
        if kernel_name == 'rbf':
            self.kernel_ = gpy.kern.RBF(input_dim=2, variance=kernel_hyps[0], lengthscale=kernel_hyps[1])
        else:
            raise ValueError('Invalid Kernel specified')

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind] = np.indices(self.dims_)
        self.grid_pts_ = np.c_[x_ind.flatten().T, y_ind.flatten().T]

    def sample_sdfs(self, num_samples=1, full_cov=True):
        """
        Samples sdfs from the GPIS
        Params:
            num_samples: (int) number of samples to generate
            full_cov: (bool) whether or not to use the diagonal or entire covariance matrix
        Returns:
            list of sdf objects
        """        
        sdf_samples = self.gp_.posterior_samples(self.grid_pts_, num_samples)
        sdfs = []
        for i in range(num_samples):
            sdf_data = sdf_samples[:,i].reshape(self.dims_)
            sdfs.append(sdf.Sdf2D(sdf_data, pose = self.pose_))
        return sdfs

    def predict_locations(self, points, full_cov=False):
        """
        Predicts the sdf values at the given points
        Params:
            points: (num_points x 2 numpy array) points to predict
            full_cov: (bool) whether or not to use the diagonal or entire covariance matrix
        Returns:
            numpy array (num_pts x 1) containing predictions
        """        
        return self.gp_.predict(points, full_cov=full_cov)

    def mean_sdf(self):
        """
        Returns an SDF of the mean surface
        Params: none
        Returns:
            SDF: mean surface
        """
        return sdf.Sdf2D(self.mean_pred_.reshape(self.dims_))

    def surface_points(self, surface_thresh = sdf.DEF_SURFACE_THRESH):
        """
        Returns the points on the surface of the mean sdf
        Params: (float) sdf value to threshold
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        return self.mean_sdf().surface_points()

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

    def scatter(self, surface_thresh = sdf.DEF_SURFACE_THRESH):
        """
        Plots the GPIS mean shape as a matplotlib 2D scatter plot, and displays the figure
        Params: - 
        Returns: - 
        """
        h = plt.figure()
        ax = h.add_subplot(111)

        # get the points on the surface
        surface_points, surface_vals = self.mean_sdf().surface_points(surface_thresh)
        x = surface_points[:,0]
        y = surface_points[:,1]

        # scatter the surface points
        ax.scatter(x, y, cmap="Blues")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, self.dims_[0])
        ax.set_ylim(0, self.dims_[1])
        plt.show()


    def imshow(self):
        """
        Displays the SDF as an image
        """
        plt.figure()
        plt.imshow(self.mean_pred_.reshape(self.dims_))
        plt.scatter(self.pts_[:,1], self.pts_[:,0])
        
        plt.show()

    def gpis_blur(self, shape_samples = None, scale = 4, contrast = 0.7):
        """
        GPIS blur visualization
        """
        # sample shapes if necessary
        if shape_samples is None:
            shape_samples = self.sample_sdfs(100)
           
        # sample surfaces and sum up
        num_samples = len(shape_samples)
        surface_image = np.zeros([scale*self.dims_[0], scale*self.dims_[1]])
        for i in range(num_samples):
            sdf_im = shape_samples[i].surface_image_thresh(scale=scale)
            surface_image = surface_image + sdf_im.astype(np.float)

        # sample the surfaces
        surface_image = (contrast / num_samples) * surface_image
        surface_image = scipy.ndimage.gaussian_filter(surface_image, 1)

        plt.figure()
        plt.imshow(surface_image, cmap=plt.get_cmap('Greys'))
        plt.show() 


def test_3d():
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    # get params from read in sdf
    all_points = sdf_3d.pts_
    all_sdf = sdf_3d.data_.flatten()
    num_points = all_points.shape[0]
    num_rand_samples = 5000

    # choose a random subset of points for construction
    random_ind = np.random.choice(num_points, num_rand_samples)
    rand_points = all_points[random_ind, :]
    rand_sdf = all_sdf[random_ind].reshape([num_rand_samples, 1])

    # create fp
    gp = Gpis3D(rand_points, rand_sdf, sdf_3d.dimensions())
    gp.scatter()

def test_2d():
    sdf_2d_file_name = 'data/test/sdf/medium_black_spring_clamp_optimized_poisson_texture_mapped_mesh_clean_0.csv'
    sf2 = sdf_file.SdfFile(sdf_2d_file_name)
    sdf_2d = sf2.read()

    # get params from read in sdf
    all_points = sdf_2d.pts_
    all_sdf = sdf_2d.data_.flatten()
    num_points = all_points.shape[0]
    num_rand_samples = 100

    # choose a random subset of points for construction
    random_ind = np.random.choice(num_points, num_rand_samples)
    rand_points = all_points[random_ind, :]
    rand_sdf = all_sdf[random_ind].reshape([num_rand_samples, 1])

    # create fp
    gp = Gpis2D(rand_points, rand_sdf, sdf_2d.dimensions())
    gp.imshow()

    gp.gpis_blur()
    exit(0)

    mean_surf_image = gp.mean_sdf().surface_image_thresh()
    plt.figure()
    plt.imshow(mean_surf_image, cmap=plt.get_cmap('Greys'))
    plt.show()

    num_shape_samp = 10
    sdf_samples = gp.sample_sdfs(num_shape_samp)

    plt.figure()
    for i in range(num_shape_samp):
        plt.subplot(1,num_shape_samp, i+1)
        sdf_samples[i].imshow()

    plt.show()

if __name__ == '__main__':
    test_2d()
