"""
Classes for registering objects from Dex-Net to tabletops imaged with the Dex-Net sensor.
Author: Jeff Mahler
"""
import caffe
import glob
import logging
import math
import numpy as np
import scipy.ndimage.filters as skf
import scipy.ndimage.morphology as snm
import os
import sys
import time

import cv
import cv2
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
from PIL import Image

import camera_params as cp
import database as db
import database_indexer as dbi
import experiment_config as ec
import feature_file as ff
import feature_matcher as fm
import image_processing as ip
import mayavi_visualizer as mv
import mesh
import obj_file as objf
import rendered_image as ri
import registration as reg
import similarity_tf as stf
import stp_file
import tfx

class DatabaseRegistrationResult:
    """ Struct to hold relevant output of registering an object from Dex-Net to real data """
    def __init__(self, tf_camera_obj, nearest_images, nearest_distances, registration_results, best_index, total_runtime):
        self.tf_camera_obj = tf_camera_obj
        self.nearest_images = nearest_images
        self.nearest_distances = nearest_distances
        self.best_index = best_index
        self.total_runtime = total_runtime

class TabletopRegistrationSolver:
    def __init__(self, logging_dir=None):
        self.logging_dir_ = logging_dir

    def log_to(self, logging_dir):
        self.logging_dir_ = logging_dir

    def _table_to_stp_transform(self, T_table_camera, x0_stp, x0_table, config):
        """ Compute a transformation to align the table normal with the z axis """
        # load table plane from calib
        R_obj_table = np.eye(3)
        calibration_dir = config['calibration_dir']
        R_camera_table = np.load(os.path.join(calibration_dir, 'rotation_camera_cb.npy'))

        # transform the table normal from camera to table basis
        n_table = T_table_camera.apply(R_camera_table[:,2], direction=True)
        table_tan_x_table = np.array([-n_table[1], n_table[0], 0])
        table_tan_x_table = table_tan_x_table / np.linalg.norm(table_tan_x_table)
        table_tan_y_table = np.cross(n_table, table_tan_x_table)
        t0 = R_obj_table[:,0].dot(table_tan_x_table)
        t1 = R_obj_table[:,0].dot(table_tan_y_table)
        xp = t0*table_tan_x_table + t1*table_tan_y_table
        xp = xp / np.linalg.norm(xp)
        yp = np.cross(n_table, xp)
        Rp = np.c_[xp, yp, n_table]

        # zero out x-y vals of target, since we only want the z values to match
        x0_table[0] = 0.0
        x0_table[1] = 0.0
        T_stp_table = stf.SimilarityTransform3D(pose=tfx.pose(Rp.T, -Rp.T.dot(x0_table)), from_frame='stp', to_frame='stp')        
        return T_stp_table

    def _create_query_image(self, color_im, depth_im, config, debug=False):
        """ Creates the query image for the database indexer """
        # read in params
        height, width, channels = color_im.shape
        font_size = config['font_size']
        if font_size is None:
            font_size = 15

        table_front_depth = config['table_front_depth'] # the depths of the front and rear of the table
        table_rear_depth = config['table_rear_depth']
        focal_length = config['focal_length']           # the camera focal length
        table_surface_tol = config['table_surface_tol'] # the tolerance of the table plane
        index_im_dim = config['index_im_dim'] # the dimension of the image to use for indexing
        depth_im_crop_dim = config['depth_im_crop_dim'] # the dimension of the cropped depth image
        depth_im_median_filter_dim = config['depth_im_median_filter_dim'] # the dimension of the cropped depth image
        depth_im_erosion_filter_dim = config['depth_im_erosion_filter_dim'] # the dimension of the cropped depth image
        cache_im_filename = config['cache_im_filename']
        calibration_dir = config['calibration_dir']

        # threshold the depths
        depth_im = ip.DepthImageProcessing.threshold_depth(depth_im, front_thresh=table_front_depth, rear_thresh=table_rear_depth)

        # project points into 3D
        camera_params = cp.CameraParams(height, width, focal_length)
        point_cloud = camera_params.deproject(depth_im)
        orig_point_cloud = ip.PointCloudProcessing.remove_zero_points(point_cloud)

        # TODO: replace with table transform
        # detect chessboard
        """
        corner_px = ip.ColorImageProcessing.find_chessboard(color_im, vis=debug)
        if corner_px is None:
            raise ValueError('Chessboard must be visible in color image')
        corner_px_round = np.round(corner_px).astype(np.uint16)
        corner_ind = ip.ij_to_linear(corner_px_round[:,0], corner_px_round[:,1], width)

        # fit a plane to the chessboard corners
        point_cloud_plane = point_cloud[:, corner_ind]
        n, mean_point_plane = ip.PointCloudProcessing.fit_plane(point_cloud_plane)
        """

        # load the camera calibration matrices
        R_camera_table = np.load(os.path.join(calibration_dir, 'rotation_camera_cb.npy'))
        t_camera_table = np.load(os.path.join(calibration_dir, 'translation_camera_cb.npy'))
        n = R_camera_table[:,2]
        mean_point_plane = t_camera_table

        # threshold to find objects on the table
        mean_point_plane = mean_point_plane + table_surface_tol * n.reshape(3,1)
        _, points_uninterest = ip.PointCloudProcessing.prune_points_above_plane(point_cloud, n, mean_point_plane)
        pixels_uninterest = ip.linear_to_ij(points_uninterest, width)
        depth_im[pixels_uninterest[:,1], pixels_uninterest[:,0]] = 0.0

        # crop image at center
        depth_im_crop = ip.DepthImageProcessing.crop_center(depth_im, depth_im_crop_dim, depth_im_crop_dim)
        if debug:
            plt.figure()
            plt.imshow(depth_im_crop, cmap=plt.cm.Greys_r, interpolation='none')
            plt.axis('off')
            plt.title('Cropped raw depth image', fontsize=font_size)
            if self.logging_dir_ is None:
                plt.show()
            else:
                figname = 'cropped_depth_image.png'
                plt.savefig(os.path.join(self.logging_dir_, figname))

        # remove spurious points by finding the largest connected object
        binary_im = ip.DepthImageProcessing.depth_to_binary(depth_im_crop)
        binary_im_ch = ip.BinaryImageProcessing.prune_contours(binary_im)
        depth_im_crop = ip.DepthImageProcessing.mask_binary(depth_im_crop, binary_im_ch)

        # filter
        depth_im_crop = skf.median_filter(depth_im_crop, size=depth_im_median_filter_dim)
        binary_mask = 1.0 * snm.binary_erosion(depth_im_crop, structure=np.ones((depth_im_erosion_filter_dim, depth_im_erosion_filter_dim)))
        depth_im_crop = ip.DepthImageProcessing.mask_binary(depth_im_crop, binary_mask)

        # center nonzero depth
        depth_im_crop_tf, diff_px = ip.DepthImageProcessing.center_nonzero_depth(depth_im_crop)
        if debug:
            plt.figure()
            plt.imshow(depth_im_crop_tf, cmap=plt.cm.Greys_r, interpolation='none')
            plt.axis('off')                            
            plt.title('Cropped, centered, and filtered depth image', fontsize=font_size)
            if self.logging_dir_ is None:
                plt.show()
            else:
                figname = 'cropped_and_centered_depth_image.png'
                plt.savefig(os.path.join(self.logging_dir_, figname))

        # compute normals for registration
        camera_c_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], focal_length,
                                          cx=depth_im_crop.shape[0]/2.0, cy=depth_im_crop.shape[1]/2.0)
        camera_i_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], focal_length,
                                          cx=index_im_dim/2.0, cy=index_im_dim/2.0)
        query_object_normals, query_point_cloud = ip.DepthImageProcessing.compute_normals(depth_im_crop_tf, camera_c_params)

        # create CNN-sized depth image for indexing
        depth_im_crop_tf_index = ip.DepthImageProcessing.crop_center(depth_im_crop_tf, index_im_dim, index_im_dim)
        binary_im_crop_tf_index = ip.DepthImageProcessing.depth_to_binary(depth_im_crop_tf_index)
        grayscale_im_crop_tf_index = ip.BinaryImageProcessing.binary_to_grayscale(binary_im_crop_tf_index)
    
        # yes, we actually need to save the image to get the JPEG compression artifacts... otherwise the nets suck 
        grayscale_im_crop_tf_index = Image.fromarray(grayscale_im_crop_tf_index)
        grayscale_im_crop_tf_index.save(cache_im_filename, 'JPEG')
        grayscale_im_crop_tf_index = np.array(Image.open(cache_im_filename).convert('RGB'))

        # find transform relative to the original image and construct query
        tf_camera_c_camera_p = ip.DepthImageProcessing.image_shift_to_transform(depth_im_crop_tf, depth_im_crop, camera_c_params, -diff_px)
        query_image = ri.RenderedImage(grayscale_im_crop_tf_index, np.zeros(3), np.zeros(3), np.zeros(3))

        if debug:
            plt.figure()
            plt.imshow(grayscale_im_crop_tf_index, cmap=plt.cm.Greys_r, interpolation='none')
            plt.axis('off')
            plt.title('Grayscale image for indexing', fontsize=font_size)
            if self.logging_dir_ is None:
                plt.show()
            else:
                figname = 'query_binary_image.png'
                plt.savefig(os.path.join(self.logging_dir_, figname))
        return query_image, query_point_cloud, query_object_normals, camera_i_params, tf_camera_c_camera_p

    def _query_database(self, query_image, database_indexer, config, debug=False):
        """ Query the database for the nearest neighbors """
        # read params
        num_nearest_neighbors = config['num_nearest_neighbors']

        # look up nearest neighbors
        nearest_neighbors = database_indexer.k_nearest(query_image, k=num_nearest_neighbors)
        nearest_images = nearest_neighbors[0]
        nearest_distances = nearest_neighbors[1]    
        
        # visualize nearest neighbors for debugging
        if debug:
            font_size = config['font_size']
            if font_size is None:
                font_size = 15

            plt.figure()
            plt.subplot(2, num_nearest_neighbors, math.ceil(float(num_nearest_neighbors)/2))
            plt.imshow(query_image.image, cmap=plt.cm.Greys_r, interpolation='none')
            plt.title('QUERY IMAGE', fontsize=font_size)
            plt.axis('off')

            for j, (image, distance) in enumerate(zip(nearest_images, nearest_distances)):
                plt.subplot(2, num_nearest_neighbors, j+num_nearest_neighbors+1)
                plt.imshow(image.image, cmap=plt.cm.Greys_r, interpolation='none')
                plt.title('NEIGHBOR %d (%d), DISTANCE = %f' %(j, image.id, distance), fontsize=font_size)
                plt.axis('off')

            if self.logging_dir_ is None:
                plt.show()
            else:
                figname = 'query_nearest_neighbors.png'
                plt.savefig(os.path.join(self.logging_dir_, figname))
        return nearest_images, nearest_distances

    def _find_best_transformation(self, query_point_cloud, query_normals, candidate_rendered_images, dataset, T_camera_p_camera_c, config, debug=False):
        """ Finds the best transformation from the candidate set using Point to Plane Iterated closest point """
        # read params
        icp_sample_size = config['icp_sample_size']
        icp_relative_point_plane_cost = config['icp_relative_point_plane_cost']
        icp_regularization_lambda = config['icp_regularization_lambda']
        feature_matcher_dist_thresh = config['feature_matcher_dist_thresh']
        feature_matcher_norm_thresh = config['feature_matcher_norm_thresh']
        num_registration_iters = config['num_registration_iters']
        compute_total_cost = config['compute_total_registration_cost']

        # register from nearest images
        registration_results = []
        min_cost = np.inf
        best_reg = None
        best_T_stp_stp_p = None
        best_index = -1
        for i, neighbor_image in enumerate(candidate_rendered_images):
            # load object mesh
            obj = dataset.graspable(neighbor_image.obj_key)

            # subdivide the mesh
            object_mesh, _ = obj.mesh.subdivide(min_triangle_length=0.025)

            # form transforms
            T_stp_camera = neighbor_image.stp_to_camera_transform().inverse()
            T_stp_obj = neighbor_image.object_to_stp_transform()
            source_mesh_ref_point = T_stp_obj.apply(neighbor_image.stable_pose.x0)

            # get source object points
            source_object_mesh = object_mesh.transform(T_stp_obj)
            mn, mx = source_object_mesh.bounding_box()
            z = mn[2]
            x0_stp = np.array([0,0,-z])

            T_stp_obj = stf.SimilarityTransform3D(pose=tfx.pose(T_stp_obj.rotation, x0_stp),
                                                  from_frame='obj', to_frame='stp')
            source_object_mesh = object_mesh.transform(T_stp_obj)
            source_object_mesh.compute_normals()
            source_object_points = np.array(source_object_mesh.vertices())
            source_object_normals = np.array(source_object_mesh.normals())

            # match table normals
            target_object_points = ip.PointCloudProcessing.remove_zero_points(query_point_cloud).T
            target_object_normals = ip.NormalCloudProcessing.remove_zero_normals(query_normals).T
            target_object_points = T_stp_camera.apply(target_object_points.T).T
            target_object_normals = T_stp_camera.apply(target_object_normals.T, direction=True).T

            calibration_dir = config['calibration_dir']
            t_camera_table = np.load(os.path.join(calibration_dir, 'translation_camera_cb.npy'))
            x0_table = T_stp_camera.apply(T_camera_p_camera_c.inverse().apply(t_camera_table))
            x0_table[2] = x0_table[2] - config['chessboard_thickness'] 

            min_z_ind = np.where(target_object_points[:,2] == np.min(target_object_points[:,2]))[0][0]
            x0_lowest = target_object_points[min_z_ind,:].T
            x0_table[2] = min(x0_table[2], x0_lowest[2])

            T_stp_stp_p = self._table_to_stp_transform(T_stp_camera, -x0_stp, x0_table, config)
            target_object_points = T_stp_stp_p.apply(target_object_points.T).T
            target_object_normals = T_stp_stp_p.apply(target_object_normals.T, direction=True).T
            x0_table2 = T_stp_stp_p.apply(x0_table)

            # display the points relative to one another
            if debug:
                subsample_inds2 = np.arange(source_object_points.shape[0])[::20]
                subsample_inds = np.arange(target_object_points.shape[0])[::20]
                T_table_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='world', to_frame='table')
                mlab.figure()                
                mlab.points3d(source_object_points[subsample_inds2,0], source_object_points[subsample_inds2,1], source_object_points[subsample_inds2,2], color=(1,0,0), scale_factor = 0.005)
                mlab.points3d(target_object_points[subsample_inds,0], target_object_points[subsample_inds,1], target_object_points[subsample_inds,2], color=(0, 1,0), scale_factor = 0.005)
                mlab.points3d(x0_table[0], x0_table[1], x0_table[2], color=(1, 1,0), scale_factor = 0.015)
                mlab.points3d(x0_table2[0], x0_table2[1], x0_table2[2], color=(0, 1,1), scale_factor = 0.015)
                mlab.points3d(0,0,0, color=(1, 1,1), scale_factor = 0.03)

                """
                t = 1e-2
                pair = np.zeros([2,3])
                for k in subsample_inds2.tolist():
                    pair[0,:] = source_object_points[k,:]
                    pair[1,:] = source_object_points[k,:] + t * source_object_normals[k,:]
                    mlab.plot3d(pair[:,0], pair[:,1], pair[:,2], color=(0,0,1), line_width=0.1, tube_radius=None)        
                """

                T_obj_world = mv.MayaviVisualizer.plot_stable_pose(object_mesh, neighbor_image.stable_pose, T_table_world)
                mlab.axes()
                mlab.show()

            # point to plane ICP solver
            ppis = reg.PointToPlaneICPSolver(sample_size=icp_sample_size, gamma=icp_relative_point_plane_cost, mu=icp_regularization_lambda)
            ppfm = fm.PointToPlaneFeatureMatcher(dist_thresh=feature_matcher_dist_thresh, norm_thresh=feature_matcher_norm_thresh) 
            registration = ppis.register_2d(source_object_points, target_object_points, source_object_normals, target_object_normals, ppfm, num_iterations=num_registration_iters,
                                            compute_total_cost=compute_total_cost, vis=debug)
            registration_results.append(registration)

            logging.info('Neighbor %d registration cost %f' %(i, registration.cost))
            if registration.cost < min_cost:
                min_cost = registration.cost
                best_reg = registration
                best_T_stp_camera = T_stp_stp_p.dot(T_stp_camera)
                best_T_stp_obj = T_stp_obj
                best_T_stp_stp_p = stf.SimilarityTransform3D(pose=tfx.pose(best_reg.R, best_reg.t), from_frame='stp', to_frame='stp')
                best_index = i

        # compute best transformation from object to camera basis
        #best_T_obj_stp = candidate_rendered_images[best_index].object_to_stp_transform().inverse()
        best_T_obj_camera = best_T_stp_obj.inverse().dot(best_T_stp_stp_p.inverse()).dot(best_T_stp_camera)

        """
        subsample_inds = np.arange(target_object_points.shape[0])[::20]
        mlab.figure()                
        target_object_points = ip.PointCloudProcessing.remove_zero_points(query_point_cloud).T
        #target_object_points = best_T_obj_camera.apply(target_object_points.T).T
        object_mesh_tf = object_mesh.transform(best_T_obj_camera.inverse())
        source_object_points = np.array(object_mesh_tf.vertices())
        mlab.points3d(source_object_points[:,0], source_object_points[:,1], source_object_points[:,2], color=(1,0,0), scale_factor = 0.005)
        mlab.points3d(target_object_points[subsample_inds,0], target_object_points[subsample_inds,1], target_object_points[subsample_inds,2], color=(0, 1,0), scale_factor = 0.005)
        mlab.show()
        """

        #best_tf_obj_camera = candidate_rendered_images[best_index].camera_to_object_transform()
        #best_tf_obj_camera_p = best_tf_camera_camera_p.pose.matrix.dot(best_tf_obj_camera.pose.matrix)
        #best_tf_obj_camera_p = stf.SimilarityTransform3D(pose=tfx.pose(best_tf_obj_camera_p))
        return best_T_obj_camera.inverse(), registration_results, best_index

    def register(self, color_im, depth_im, dataset, database_indexer, config, debug=False):
        """
        Register an object from Dex-Net to the tabletop in the color and depth images.
        The conventions are:
           obj = object basis
           camera_c = virtual cropped image camera basis
           camera_p = real world camera basis
        Params:
           color_im: (HxWx3 numpy uint8 array) The color image corresponding to the scene to register
           depth_im: (HxW numpy float array) The depth image corresponding to the scene to register. Must be in same frame of reference as the color image
           dataset:  (Hdf5Dataset instance) Dataset containing the data
           database_indexer: (Hdf5DatabaseIndexer instance) Object to index Dex-Net for registration hypotheses
           config: (ExperimentConfig or python dictionary) Key-value dictionary-like object with the following params:
              - table_front_depth (float) - depth at the front of the table
              - table_rear_depth  (float) - depth at the rear of the table
              - table_surface_tol    (float) - distance in millimeters above the table plane to crop the image
              - focal_length         (float) - focal length of the camera in pixels
              - index_im_dim         (int)   - dimension of the image to use for indexing (256)
              - depth_im_crop_dim    (int)   - dimension of the initial depth crop (for removing the table and spurious points)
              - depth_im_median_filter_dim (int)  - window size of median filter
              - depth_im_erosion_filter_dim (int) - window size of erosion filter
              - cache_im_filename     (string) - filename to save preprocessed query image
              - num_nearest_neighbors (int)    - number of nearest neighbors to use for registration
              - icp_sample_size       (int)    - number of subsampled points to use in ICP iterations (similar to batch size in SGD)
              - icp_relative_point_plane_cost (float) - multiplicative factor for point-based ICP cost term relative to point-plane ICP cost term
              - icp_regularization_lambda     (int)   - regularization constant for ICP
              - feature_matcher_dist_thresh   (float) - point distance threshold for ICP feature matching
              - feature_matcher_norm_thresh   (float) - normal inner product threshold for ICP feature matching
              - num_registration_iters        (int)   - number of iterations to run registration for
           debug: (bool) whether or not to display additional debugging output
        Returns:
           DatabaseRegistrationResult object
        """
        if color_im.shape[0] != depth_im.shape[0] or color_im.shape[1] != depth_im.shape[1]:
            raise ValueError('Color and depth images must have the same dimension')

        debug_or_save = debug or (self.logging_dir_ is not None)

        # remove points beyond table
        ip_start = time.time()
        query_image, query_point_cloud, query_normals, camera_c_params, T_camera_p_camera_c = \
            self._create_query_image(color_im, depth_im, config, debug=debug_or_save)
        ip_end = time.time()

        # index the database for similar objects
        query_start = time.time()
        nearest_images, nearest_distances = self._query_database(query_image, database_indexer, config, debug=debug_or_save)
        query_end = time.time()

        # register to the candidates
        registration_start = time.time()
        T_camera_c_obj, registration_results, best_index = \
            self._find_best_transformation(query_point_cloud, query_normals, nearest_images, dataset, T_camera_p_camera_c, config, debug=debug)
        T_camera_p_obj = T_camera_p_camera_c.dot(T_camera_c_obj)
        registration_end = time.time()

        # log runtime
        total_runtime = registration_end-registration_start + query_end-query_start + ip_end-ip_start
        logging.info('Image processing took %.2f sec' %(ip_end-ip_start))
        logging.info('Database query took %.2f sec' %(query_end-query_start))
        logging.info('ICP took %.2f sec' %(registration_end-registration_start))
        logging.info('Total registration time: %.2f sec' %(total_runtime))

        # display transformed mesh projected into the query image
        if debug:
            font_size = config['font_size']
            if font_size is None:
                font_size = 15

            best_object = dataset.graspable(nearest_images[best_index].obj_key)
            best_object_mesh = best_object.mesh
            best_object_mesh_tf = best_object_mesh.transform(T_camera_c_obj)
            object_point_cloud = np.array(best_object_mesh_tf.vertices()).T
            object_mesh_proj_pixels, mesh_valid_ind = camera_c_params.project(object_point_cloud)

            plt.figure()
            plt.imshow(query_image.image, cmap=plt.cm.Greys_r, interpolation='none')
            plt.scatter(object_mesh_proj_pixels[0,mesh_valid_ind], object_mesh_proj_pixels[1,mesh_valid_ind], s=80, c='r')
            plt.title('Projected object mesh pixels', fontsize=font_size)
            if self.logging_dir_ is None:
                plt.show()
            else:
                figname = 'projected_mesh.png'
                plt.savefig(os.path.join(self.logging_dir_, figname))

        # construct and return output
        return DatabaseRegistrationResult(T_camera_p_obj, nearest_images, nearest_distances, registration_results, best_index, total_runtime)

class KnownObjectTabletopRegistrationSolver(TabletopRegistrationSolver):
    def __init__(self, object_key, dataset, config, logging_dir=None):
        TabletopRegistrationSolver.__init__(self, logging_dir)
        self.object_key_ = object_key
        self.dataset_ = dataset
        self.cnn_indexer_ = dbi.CNN_Hdf5ObjectIndexer(object_key, dataset, config)
        self.config_ = config

    def register(self, color_im, depth_im, debug=False):
        """ Create a CNN object indexer for registration """
        return TabletopRegistrationSolver.register(self, color_im, depth_im, self.dataset_, self.cnn_indexer_, self.config_['registration'], debug=debug)

class KnownObjectStablePoseTabletopRegistrationSolver(TabletopRegistrationSolver):
    def __init__(self, object_key, stable_pose_id, dataset, config, logging_dir=None):
        TabletopRegistrationSolver.__init__(self, logging_dir)
        self.object_key_ = object_key
        self.dataset_ = dataset
        self.cnn_indexer_ = dbi.CNN_Hdf5ObjectStablePoseIndexer(object_key, stable_pose_id, dataset, config)
        self.config_ = config

    def register(self, color_im, depth_im, debug=False):
        """ Create a CNN object indexer for registration """
        return TabletopRegistrationSolver.register(self, color_im, depth_im, self.dataset_, self.cnn_indexer_, self.config_['registration'], debug=debug)
