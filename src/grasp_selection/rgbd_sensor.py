from primesense import openni2
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
import mesh
import obj_file as objf
import rendered_image as ri
import registration as reg
import similarity_tf as stf
import stp_file
import tabletop_object_registration as tor
import tfx

CHANNELS = 3
OPENNI2_PATH = '/home/jmahler/Libraries/OpenNI-Linux-x64-2.2/Redist'

class RgbdSensor(object):
    """
    Crappy RGBD sensor class. Can't do 30 fps or anywhere close, but can at least return color & depth images
    """
    def __init__(self, width=640, height=480, fps=30, path_to_ni_lib=OPENNI2_PATH, intrinsics=None, auto_start=True):
        openni2.initialize(path_to_ni_lib)
        self.width_ = width
        self.height_ = height
        self.fps_ = fps
        self.dev_ = openni2.Device.open_any()
        self.depth_stream_ = None
        self.color_stream_ = None

        self._configure()

        if auto_start:
            self.start()

    def _configure(self):
        pass
        #self.dev_.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        
    def start(self):
        self.depth_stream_ = self.dev_.create_depth_stream()
        self.depth_stream_.configure_mode(self.width_, self.height_, self.fps_, openni2.PIXEL_FORMAT_DEPTH_1_MM) 
        self.depth_stream_.start()

        self.color_stream_ = self.dev_.create_color_stream()
        self.color_stream_.configure_mode(self.width_, self.height_, self.fps_, openni2.PIXEL_FORMAT_RGB888) 
        #self.color_stream_.camera.set_auto_white_balance(False)
        #self.color_stream_.camera.set_auto_exposure(False)
        self.color_stream_.start()

        self.dev_.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    def stop(self):
        if self.depth_stream_:
            self.depth_stream_.stop()
        if self.color_stream_:
            self.color_stream_.stop()

    def reset(self):
        self.stop()
        self.start()

    def get_depth_image(self):
        frame = self.depth_stream_.read_frame()
        raw_buf = frame.get_buffer_as_uint16()
        buf_array = np.array([raw_buf[i] for i in range(self.width_ * self.height_)])
        depth_image = buf_array.reshape(self.height_, self.width_)
        depth_image = depth_image * 0.001 # convert to meters
        return np.fliplr(depth_image)

    def get_color_image(self):
        frame = self.color_stream_.read_frame()
        raw_buf = frame.get_buffer_as_triplet()
        r_array = np.array([raw_buf[i][0] for i in range(self.width_ * self.height_)])        
        g_array = np.array([raw_buf[i][1] for i in range(self.width_ * self.height_)])        
        b_array = np.array([raw_buf[i][2] for i in range(self.width_ * self.height_)])        
        color_image = np.zeros([self.height_, self.width_, CHANNELS])
        color_image[:,:,0] = r_array.reshape(self.height_, self.width_)
        color_image[:,:,1] = g_array.reshape(self.height_, self.width_)
        color_image[:,:,2] = b_array.reshape(self.height_, self.width_)
        return np.fliplr(color_image.astype(np.uint8))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    config_filename = sys.argv[1]
    object_key = sys.argv[2]
    logging.info('Registering to object %s' %(object_key))

    # params
    load = False
    save = False
    debug = True
    vis = True
    eps = 0.0025
    table_end_depth = 1.0
    focal_length = 525.
    cnn_dim = 256
    depth_im_crop_dim = 300
    num_avg_images = 1

    depth_im_median_filter_dim = 9.0
    depth_im_erosion_filter_dim = 3
    num_nearest_neighbors = 5
    num_registration_iters = 5

    feature_matcher_dist_thresh = 0.05
    feature_matcher_norm_thresh = 0.75
    icp_sample_size = 100
    icp_relative_point_plane_cost = 100.0
    icp_regularizatiion_lambda = 1e-2

    font_size = 15
    tube_radius = 0.002

    cache_image_filename = 'data/cnn_grayscale_image.jpg'
    
    # open up the config and database
    config = ec.ExperimentConfig(config_filename)
    database_name = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_name, config)
    dataset = database.dataset(config['datasets'].keys()[0])

    # create sensor
    s = RgbdSensor()
    if load:
        f = open('data/test/rgbd/depth_im.npy', 'r')
        depth_im = np.load(f)
        f = open('data/test/rgbd/color_im.npy', 'r')
        color_im = np.load(f)
        f = open('data/test/rgbd/corners.npy', 'r')
        corner_px = np.load(f)
    else:
        # average a bunch of depth images together
        depth_im = np.zeros([s.height_, s.width_])
        counts = np.zeros([s.height_, s.width_])
        for i in range(num_avg_images):
            new_depth_im = s.get_depth_image()

            depth_im = depth_im + new_depth_im
            counts = counts + np.array(new_depth_im > 0.0)

        # retrieve and store images
        depth_im[depth_im > 0] = depth_im[depth_im > 0] / counts[depth_im > 0]
        if save:
            f = open('data/test/rgbd/depth_im.npy', 'w')
            np.save(f, depth_im)

        color_im = s.get_color_image()
        if save:
            f = open('data/test/rgbd/color_im.npy', 'w')
            np.save(f, color_im)

        # find the chessboard
        corner_px = ip.ColorImageProcessing.find_chessboard(color_im, vis=False)
        if save:
            f = open('data/test/rgbd/corners.npy', 'w')
            np.save(f, corner_px)

    if vis:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(color_im)
        plt.axis('off')
        plt.title('Original Color Image', fontsize=font_size)
        plt.subplot(1,2,2)
        plt.imshow(depth_im, cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.title('Original Depth Image', fontsize=font_size)
        plt.show()

    # project depth image to get point cloud
    depth_im = ip.DepthImageProcessing.threshold_depth(depth_im, rear_thresh=config['registration']['table_rear_depth'])
    camera_params = cp.CameraParams(s.height_, s.width_, config['registration']['focal_length'])
    point_cloud = camera_params.deproject(depth_im)
    orig_point_cloud = ip.PointCloudProcessing.remove_zero_points(point_cloud)

    # find the known object for registration
    kotor = tor.KnownObjectTabletopRegistrationSolver()
    reg_result = kotor.register(color_im, depth_im, object_key, dataset, config, debug=debug)

    if vis:
        # read out parameters
        tf_obj_camera_p = reg_result.tf_camera_obj
        obj = dataset.graspable(object_key)
        object_mesh = obj.mesh
        object_mesh_tf= object_mesh.transform(tf_obj_camera_p)

        # transform mesh
        object_mesh_tf = object_mesh.transform(tf_obj_camera_p)
        source_object_points = np.array(object_mesh_tf.vertices())
        target_object_points = orig_point_cloud.T
        subsample_inds = np.arange(target_object_points.shape[0])[::20]

        # load grasps
        metric = 'pfc_f_0.200000_tg_0.020000_rg_0.020000_to_0.020000_ro_0.020000'
        sorted_grasps, sorted_metrics = dataset.sorted_grasps(object_key, metric)
        best_grasp = sorted_grasps[0]
        g1, g2 = best_grasp.endpoints()
        grasp_points = np.c_[np.array(g1), np.array(g2)]
        grasp_points_tf = tf_obj_camera_p.apply(grasp_points)
        grasp_points_tf = grasp_points_tf.T

        mlab.figure()
        object_mesh_tf.visualize(color=(1,0,0), style='wireframe')
        mlab.points3d(target_object_points[subsample_inds,0], target_object_points[subsample_inds,1], target_object_points[subsample_inds,2], color=(0, 1,0), scale_factor = 0.005)
        mlab.plot3d(grasp_points_tf[:,0], grasp_points_tf[:,1], grasp_points_tf[:,2], color=(0,0,1), tube_radius=tube_radius)
        mlab.points3d(grasp_points_tf[:,0], grasp_points_tf[:,1], grasp_points_tf[:,2], color=(0,0,1), scale_factor = 0.02)
        mlab.title('Object Mesh (Red) Overlayed \n on Point Cloud (Green)', size=font_size)
        mlab.show()

        tf_obj_camera_p.save('data/calibration/spray_pose.stf')
    IPython.embed()
