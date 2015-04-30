""" 
Definition of SDF Class
Author: Sahaana Suri & Jeff Mahler

**Currently assumes clean input**
"""
from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
import scipy.io
import scipy.ndimage
import scipy.signal
from skimage import feature
import skimage.filters

import sdf_file as sf
import tfx
import time

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import IPython

from sys import version_info
if version_info[0] != 3:
    range = xrange

MAX_CHAR = 255
DEF_SURFACE_THRESH = 0.05

def crosses_threshold(threshold):
    def crosses(elems):
        """ 
        Function to determine if a np array has values both above and below a threshold (moved edge). Generalized "has positive and negative" function. 
        For use with filter(-,-). 
        Params: 
            elems: np array of numbers
            threshold: any number used as a threshold
        Returns: 
            (bool): True if elems has both negative and positive numbers, False otherwise
        """
        return (elems>threshold).any() and (elems<threshold).any()
    return crosses


class Sdf:
    __metaclass__ = ABCMeta

    def dimensions(self):
        """ 
        SDF dimension information
        Returns: 
            numpy 2 or 3 array: the dimensions of the sdf
        """
        return self.dims_

    def origin(self):
        """
        Object origin
        Returns:
            numpy 2 or 3 array: the object's origin
        """
        return self.origin_

    def resolution(self):
        """
        Object resolution (how wide each grid cell is)
        Returns:
            float: object resolution
        """
        return self.resolution_

    def data(self):
        """
        Returns the SDF data
        Returns:
            numpy 2 or 3 array: sdf array
        """
        return self.data_

    def pose(self):
        """
        Returns the pose of the sdf wrt world frame
        Returns:
            tfx pose: sdf pose
        """
        return self.pose_

    @abstractmethod
    def __getitem__(self, coords):
        """
        Returns the signed distance at the given coordinates
        Params: numpy 2 or 3 array
        Returns:
            float: the signed distance and the given coorss (interpolated)
        """
        pass

    @abstractmethod
    def surface_points(self, surf_thresh):
        """
        Returns the points on the surface
        Params: (float) sdf value to threshold
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        pass        

class Sdf3D(Sdf):
    def __init__(self, sdf_data, origin, resolution, pose = tfx.identity_tf(frame="world"), use_abs = True):
        self.data_ = sdf_data
        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape
        self.pose_ = pose
        self.center_ = [self.dims_[0] / 2, self.dims_[1] / 2, self.dims_[2] / 2]

        # optionally use only the absolute values (useful for non-closed meshes in 3D)
        if use_abs:
            self.data_ = np.abs(self.data_)

        self._compute_flat_indices()

        self.feature_vector_ = None #Kmeans feature representation

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind, z_ind] = np.indices(self.dims_)
        self.pts_ = np.c_[x_ind.flatten().T, np.c_[y_ind.flatten().T, z_ind.flatten().T]]

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
        new_coords[0] = max(0, min(coords[0], self.dims_[0] - 1))
        new_coords[1] = max(0, min(coords[1], self.dims_[1] - 1))
        new_coords[2] = max(0, min(coords[2], self.dims_[2] - 1))

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

    def surface_points(self, surface_thresh = DEF_SURFACE_THRESH):
        """
        Returns the points on the surface
        Params: (float) sdf value to threshold
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        surface_points = np.where(np.abs(self.data_) < surface_thresh)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_vals = self.data_[surface_points[:,0], surface_points[:,1], surface_points[:,2]]
        return surface_points, surface_vals

    def transform(self, T, scale = 1.0, detailed = False):
        """
        Transform the grid by pose T and scale with canonical reference frame at the SDF center with axis alignment
        Params:
            (tfx pose): Pose T
            (float): scale of transform
            (boo): detailed - whether or not to do the dirty, fast method
        Returns:
            (SDF): new sdf with grid warped by T
        """
        # map all surface points to their new location
        start_t = time.clock()
        num_pts = self.pts_.shape[0]
        pts_centered = self.pts_ - self.center_
        pts_homog = np.r_[pts_centered.T, np.ones([1, num_pts])]
        pts_homog_tf = T.matrix.dot(pts_homog)
        pts_tf_centered = (1.0 / scale) * pts_homog_tf[0:3,:].T
        pts_tf = pts_tf_centered + self.center_
        all_points_t = time.clock()

        # transform the center
        origin_centered = self.origin_ - self.center_
        origin_homog = np.r_[origin_centered.T, 1]
        origin_homog_tf = T.matrix.dot(origin_homog)
        origin_tf_centered = (1.0 / scale) * origin_homog_tf[0:3,:].T
        origin_tf = origin_tf_centered + self.center_

        # rescale the resolution
        resolution_tf = scale * self.resolution_
        origin_res_t = time.clock()

        # add each point to the new pose
        if detailed:
            sdf_data_tf = np.zeros([num_pts, 1])
            for i in range(num_pts):
                sdf_data_tf[i] = self[pts_tf[i,0], pts_tf[i,1], pts_tf[i,2]]
        else:
            pts_tf_round = np.round(pts_tf).astype(np.int64)
            
            # snap to closest boundary
            pts_tf_round[:,0] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,0]], axis=1)
            pts_tf_round[:,0] = np.min(np.c_[(self.dims_[0] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,0]], axis=1)

            pts_tf_round[:,1] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,1]], axis=1)
            pts_tf_round[:,1] = np.min(np.c_[(self.dims_[1] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,1]], axis=1)

            pts_tf_round[:,2] = np.max(np.c_[np.zeros([num_pts, 1]), pts_tf_round[:,2]], axis=1)
            pts_tf_round[:,2] = np.min(np.c_[(self.dims_[2] - 1) * np.ones([num_pts, 1]), pts_tf_round[:,2]], axis=1)

            sdf_data_tf = self.data_[pts_tf_round[:,0], pts_tf_round[:,1], pts_tf_round[:,2]]
            
        sdf_data_tf_grid = sdf_data_tf.reshape(self.dims_)
        tf_t = time.clock()

        logging.info('Sdf3D: Time to transform coords: %f' %(all_points_t - start_t))
        logging.info('Sdf3D: Time to transform origin: %f' %(origin_res_t - all_points_t))
        logging.info('Sdf3D: Time to transfer sd: %f' %(tf_t - origin_res_t))

        return Sdf3D(sdf_data_tf_grid, origin_tf, resolution_tf, pose = T * self.pose_)

    def make_windows(self, W, S, target=False, filtering_function=crosses_threshold, threshold=.1): 
        """ 
        Function for windowing the SDF grid
        Params: 
            W: the side length of the window (currently assumed to be odd so centering makes sense)
            S: stride length between cubes (x axis "wraps" around)
            target: True for targetted windowing (filters for cubes containing both positive and negative values)
            filtering_function: function to filter windows out with
        Returns: 
            ([np.array,...,np.array]): contains a list of numpy arrays, each of which is an unrolled window/cube.
                                       window order based on center coordinate (increasing order of x, y, then z)
        """
        windows = []
        window_center = (W-1)/2 #assuming odd window
        offset = 0 #parameter used to "wrap around" x / not start back at x=1 each time  

        nx = self.dims_[0]
        ny = self.dims_[1]
        nz = self.dims_[2]
        newx = 2*window_center + nx
        newy = 2*window_center + ny
        newz = 2*window_center + nz
        padded_vals = np.zeros(newx*newy*newz)
    
        #padding the original values list 
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    padded_coord = (i+window_center) + (j+window_center)*newx + (k+window_center)*newx*newy
                    orig_coord = i + j*nx + k*nx*ny
                    padded_vals[padded_coord] = self.values_in_order_[orig_coord]

        for z in range(window_center, nz + window_center, S):
            for y in range(window_center, ny + window_center, S):
                for x in range(window_center+offset, nx + window_center, S):
                        #print map(lambda x: x-window_center,[x,y,z])
                    new_window = np.zeros(W**3) 
                    count = 0
                    for k in range(-window_center, window_center+1):    
                        for j in range(-window_center, window_center+1):
                            for i in range(-window_center, window_center+1):
                                new_window[count] = padded_vals[(x+i) + (y+j)*newx + (z+k)*newx*newy] 
                                count += 1
                    windows.append(new_window)
                offset = (x+S) - (nx+window_center)
        #print windows, len(windows), type(windows)
        if target:
            windows = filter(filtering_function(threshold), windows)          
        return windows       

    def set_feature_vector(self, vector):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor
        """
        self.feature_vector_ = vector

    def feature_vector(self):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor 
        """
        return self.feature_vector_
     

    def add_to_nearpy_engine(self, engine):
        """
        Inserts the SDF into the provided nearpy Engine
        Params: 
            engine: nearpy.engine.Engine 
        Returns: - 
        """
        if self.feature_vector_ is None:
            to_add = self.data_[:]
        else:
            to_add = self.feature_vector

        engine.store_vector(to_add,self.file_name_)


    def query_nearpy_engine(self, engine):
        """
        Queries the provided nearpy Engine for the SDF closest to this one
        Params:
            engine: nearpy.engine.Engine 
        Returns:
            (list (strings), list (tuples))
            list (strings): Names of the files that most closely match this one
            list (tuple): Additional information regarding the closest matches in (numpy.ndarray, string, numpy.float64) form:
                numpy.ndarray: the full vector of values of that match (equivalent to that SDF's "values_in_order_")
                string: the match's SDF's file name
                numpy.float64: the match's distance from this SDF
        """
        if self.feature_vector is None:
            to_query = self.data_[:]
        else:
            to_query = self.feature_vector
        results = engine.neighbours(to_query)
        file_names = [i[1] for i in results]
        return file_names, results

    def send_to_matlab(self, out_file):
        """
        Saves the SDF's coordinate and value information to the provided matlab workspace
        Params:
            out_file: string
        Returns: - 
        """
        # TODO: fix this
#        scipy.io.savemat(out_file, mdict={'X':self.xlist_, 'Y': self.ylist_, 'Z': self.zlist_, 'vals': self.values_in_order_})
        logging.info("SDF information saved to %s" % out_file)

    def scatter(self, surface_thresh = DEF_SURFACE_THRESH):
        """
        Plots the SDF as a matplotlib 3D scatter plot, and displays the figure
        Params: - 
        Returns: - 
        """
        h = plt.figure()
        ax = h.add_subplot(111, projection = '3d')

        # surface points
        surface_points, surface_vals = self.surface_points(surface_thresh)
        x = surface_points[:,0]
        y = surface_points[:,1]
        z = surface_points[:,2]

        # scatter the surface points
        ax.scatter(x, y, z, cmap="Blues")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(0, self.dims_[0])
        ax.set_ylim3d(0, self.dims_[1])
        ax.set_zlim3d(0, self.dims_[2])

class Sdf2D(Sdf):
    def __init__(self, sdf_data, origin = np.array([0,0]), resolution = 1.0, pose = tfx.identity_tf(frame="world")):
        self.data_ = sdf_data
        self.origin_ = origin
        self.resolution_ = resolution
        self.dims_ = self.data_.shape
        self.pose_ = pose
        self.surface_thresh_ = np.percentile(np.abs(self.data_), 5)
        self.center_ = [self.dims_[0] / 2, self.dims_[1] / 2]

        self._compute_flat_indices()

        self.feature_vector_ = None #Kmeans feature representation

    def _compute_flat_indices(self):
        """
        Gets the indices of the flattened array
        """
        [x_ind, y_ind] = np.indices(self.dims_)
        self.pts_ = np.c_[x_ind.flatten().T, y_ind.flatten().T];

    def __getitem__(self, coords):
        """
        Returns the signed distance at the given coordinates, interpolating if necessary
        Params: numpy 3 array
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
        new_coords[0] = max(0, min(coords[0], self.dims_[0]-1))
        new_coords[1] = max(0, min(coords[1], self.dims_[1]-1))

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

    def surface_points(self, surface_thresh = DEF_SURFACE_THRESH):
        """
        Returns the points on the surface
        Params: (float) sdf value to threshold
        Returns:
            numpy arr: the points on the surfaec
            numpy arr: the sdf values on the surface
        """
        surface_points = np.where(np.abs(self.data_) < self.surface_thresh_)
        x = surface_points[0]
        y = surface_points[1]
        surface_points = np.c_[x, y]
        surface_vals = self.data_[surface_points[:,0], surface_points[:,1]]
        return surface_points, surface_vals

    def surface_image_thresh(self, surface_thresh = DEF_SURFACE_THRESH, alpha = 0.5, scale = 4):
        """
        Returns an image that is zero on the shape surface and one otherwise
        """
        surface_points, surface_vals = self.surface_points(self.surface_thresh_)
        surface_image = np.zeros(self.dims_).astype(np.uint8)
        surface_image[surface_points[:,0], surface_points[:,1]] = MAX_CHAR

        # laplacian
        surface_image = np.array(Image.fromarray(surface_image).resize((scale*self.dims_[0], scale*self.dims_[1]), Image.ANTIALIAS))
        surface_image = scipy.ndimage.gaussian_filter(surface_image, 2)
        filter_blurred_l = scipy.ndimage.gaussian_filter(surface_image, 0.1)
        surface_image = surface_image + alpha * (surface_image - filter_blurred_l)

        # take only points higher than a certain value
        surface_image = MAX_CHAR * (surface_image > 70)
        return surface_image

    def transform(self, T, scale = 1.0):
        """
        Transform the grid by pose T and scale with canonical reference frame at the SDF center with axis alignment
        Params:
            (tfx pose): Pose T
            (float): scale of transform
        Returns:
            (SDF): new sdf with grid warped by T
        """
        # map all surface points to their new location
        num_pts = self.pts_.shape[0]
        pts_centered = self.pts_ - self.center_
        pts_homog = np.r_[pts_centered.T, np.r_[np.zeros([1, num_pts]), np.ones([1, num_pts])]]
        pts_homog_tf = T.matrix.dot(pts_homog)
        pts_tf_centered = (1.0 / scale) * pts_homog_tf[0:2,:].T
        pts_tf = pts_tf_centered + self.center_

        # add each point to the new pose
        sdf_data_tf = np.zeros([num_pts, 1])
        for i in range(num_pts):
            sdf_data_tf[i] = self[pts_tf[i,0], pts_tf[i,1]]
        sdf_data_tf_grid = sdf_data_tf.reshape(self.dims_)

        return Sdf2D(sdf_data_tf_grid, pose = T * self.pose_)

    def set_feature_vector(self, vector):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor
        """
        self.feature_vector_ = vector

    def feature_vector(self):
        """
        Sets the features vector of the SDF
        TODO: object oriented feature extractor 
        """
        return self.feature_vector_
     

    def add_to_nearpy_engine(self, engine):
        """
        Inserts the SDF into the provided nearpy Engine
        Params: 
            engine: nearpy.engine.Engine 
        Returns: - 
        """
        if self.feature_vector_ is None:
            to_add = self.data_[:]
        else:
            to_add = self.feature_vector

        engine.store_vector(to_add,self.file_name_)


    def query_nearpy_engine(self, engine):
        """
        Queries the provided nearpy Engine for the SDF closest to this one
        Params:
            engine: nearpy.engine.Engine 
        Returns:
            (list (strings), list (tuples))
            list (strings): Names of the files that most closely match this one
            list (tuple): Additional information regarding the closest matches in (numpy.ndarray, string, numpy.float64) form:
                numpy.ndarray: the full vector of values of that match (equivalent to that SDF's "values_in_order_")
                string: the match's SDF's file name
                numpy.float64: the match's distance from this SDF
        """
        if self.feature_vector is None:
            to_query = self.data_[:]
        else:
            to_query = self.feature_vector
        results = engine.neighbours(to_query)
        file_names = [i[1] for i in results]
        return file_names, results

    def send_to_matlab(self, out_file):
        """
        Saves the SDF's coordinate and value information to the provided matlab workspace
        Params:
            out_file: string
        Returns: - 
        """
        # TODO: fix this
#        scipy.io.savemat(out_file, mdict={'X':self.xlist_, 'Y': self.ylist_, 'Z': self.zlist_, 'vals': self.values_in_order_})
        logging.info("SDF information saved to %s" % out_file)

    def scatter(self):
        """
        Plots the SDF as a matplotlib 2D scatter plot, and displays the figure
        Params: - 
        Returns: - 
        """
        h = plt.figure()
        ax = h.add_subplot(111)

        # get the points on the surface
        surface_points, surface_vals = self.surface_points(self.surface_thresh_)
        x = surface_points[:,0]
        y = surface_points[:,1]

        # scatter the surface points
        ax.scatter(y, x, cmap="Blues")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, self.dims_[0])
        ax.set_ylim(0, self.dims_[1])

    def imshow(self):
        """
        Displays the SDF as an image
        """
        plt.imshow(self.data_ < 0, cmap=plt.get_cmap('Greys'))

    def vis_surface(self):
        """
        Displays the SDF surface image
        """
        plt.imshow(self.surface_image_thresh(), cmap=plt.get_cmap('Greys'))

def test_function():
    test_sdf = "aunt_jemima_original_syrup/processed/textured_meshes/optimized_tsdf_texture_mapped_mesh.sdf"
    matlab_file = "data.mat"
    teatime = SDF(test_sdf)
    print "Done processing %s" % test_sdf
    print "Dimension: %d, x: %d, y: %d, z: %d" % teatime.dimensions()
    print "Origin: (%f, %f, %f)" % teatime.origin()
    print "Spacing: %f" % teatime.spacing()
    print "Data Matrix: \n", teatime.data()

    #Testing LSH by just inserting the same sdf file twice, then querying it
    dimension = teatime.dimensions()[0]
    rbp = RandomBinaryProjections('rbp',10)
    engine = Engine(dimension, lshashes=[rbp])
    teatime.add_to_nearpy_engine(engine)
    teatime.add_to_nearpy_engine(engine)
    print "Query results: \n", teatime.query_nearpy_engine(engine)

    teatime.send_to_matlab(matlab_file)
    teatime.make_plot()

def test_3d_transform():
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    s = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = s.read()

    # transform
    pose_mat = np.matrix([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0,0,1]])
    tf = tfx.transform(pose_mat, frame='world')
    tf = tfx.random_tf()

    start_t = time.clock()
    sdf_tf = sdf_3d.transform(tf, scale = 1.5)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('3D Transform took %f sec' %(duration))
    logging.info('Transformed resolution %f' %(sdf_tf.resolution()))

    start_t = time.clock()
    sdf_tf_d = sdf_3d.transform(tf, scale = 1.5, detailed = True)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('Detailed 3D Transform took %f sec' %(duration))
    logging.info('Transformed detailed resolution %f' %(sdf_tf_d.resolution()))

    # display
    sdf_3d.scatter()
    plt.title('Original')
    sdf_tf.scatter()
    plt.title('Transformed')
    sdf_tf_d.scatter()
    plt.title('Detailed Transformed')
    plt.show()
    
def test_2d_transform():
    sdf_2d_file_name = 'data/test/sdf/medium_black_spring_clamp_optimized_poisson_texture_mapped_mesh_clean_0.csv'
    sf2 = sf.SdfFile(sdf_2d_file_name)
    sdf_2d = sf2.read()

    # transform
    pose_mat = np.matrix([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0,0,1]])
    tf = tfx.transform(pose_mat, frame='world')

    start_t = time.clock()
    sdf_tf = sdf_2d.transform(tf, scale = 1.5)
    end_t = time.clock()
    duration = end_t - start_t
    logging.info('2D Transform took %f sec' %(duration))

    # display
    plt.figure()
    plt.subplot(1,2,1)
    sdf_2d.imshow()
    plt.title('Original')

    plt.subplot(1,2,2)
    sdf_tf.imshow()
    plt.title('Transformed')

    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
#    test_2d_transform()
    test_3d_transform()

