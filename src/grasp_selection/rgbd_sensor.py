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
import tfx

CHANNEL_SWAP = (2, 1, 0)
CHANNELS = 3
OPENNI2_PATH = '/home/jmahler/Libraries/OpenNI-Linux-x64-2.2/Redist'

def ij_to_linear(i, j, width):
    return i + j.dot(width)

def linear_to_ij(ind, width):
    return np.c_[ind % width, ind / width]

class MVCNNBatchFeatureExtractor():
    # TODO: update to use database at some point
    def __init__(self, config):
        self.config_ = config
        self.caffe_config_ = self.config_['caffe']
        self._parse_config()
        self.net_ = self._init_cnn()

    def _parse_config(self):
        self.pooling_method_ = self.caffe_config_['pooling_method']
        self.rendered_image_ext_ = self.caffe_config_['rendered_image_ext']
        self.images_per_object_ = self.caffe_config_['images_per_object']
        self.path_to_image_dir_ = self.caffe_config_['rendered_image_dir']
        self.caffe_data_dir_ = self.caffe_config_['config_dir']
        self.batch_size_ = self.caffe_config_['batch_size']
        self.caffe_model_ = os.path.join(self.caffe_data_dir_, self.caffe_config_['caffe_model'])
        self.deploy_model_ = os.path.join(self.caffe_data_dir_, self.caffe_config_['deploy_file']) 
        self.mean_ = np.load(os.path.join(self.caffe_data_dir_, self.caffe_config_['mean_file'])).mean(1).mean(1)

    def _init_cnn(self):
        caffe.set_mode_gpu() if self.caffe_config_['deploy_mode'] == 'gpu' else caffe.set_mode_cpu()
        net = caffe.Classifier(self.deploy_model_, self.caffe_model_,
                               mean=self.mean_,
                               channel_swap=CHANNEL_SWAP,
                               raw_scale=self.caffe_config_['raw_scale'],
                               image_dims=(self.caffe_config_['image_dim_x'], self.caffe_config_['image_dim_x']))
        return net

    def _forward_pass(self, images):
        load_start = time.time()
        loaded_images = map(caffe.io.load_image, images)
        load_stop = time.time()
        logging.debug('Loading took %f sec' %(load_stop - load_start))
        final_blobs = self.net_.predict(loaded_images, oversample=False)
        fp_stop = time.time()
        logging.debug('Prediction took %f sec per image' %((fp_stop - load_stop) / len(loaded_images)))
        return final_blobs

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

def find_chessboard(raw_image, sx=6, sy=9):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((sx*sy,3), np.float32)
    objp[:,:2] = np.mgrid[0:sx,0:sy].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # create images
    img = raw_image.astype(np.uint8)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (sx,sy),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_rgb, (sx,sy), corners, ret)
        cv2.imshow('img',img_rgb)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
    if corners is not None:
        return corners.squeeze()
    return None

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    config_filename = sys.argv[1]
    object_key = sys.argv[2]
    logging.info('Registering to object %s' %(object_key))

    # params
    load = False
    save = False
    debug = False
    vis = True
    eps = 0.01
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
    font_size = 15

    cache_image_filename = 'data/cnn_grayscale_image.jpg'
    
    icp_sample_size = 100
    icp_relative_point_plane_cost = 100.0
    icp_regularizatiion_lambda = 1e-2

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
        corner_px = ip.ColorImageProcessing.find_chessboard(color_im)
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

    # remove points beyond table
    depth_im = ip.DepthImageProcessing.threshold_depth(depth_im, table_end_depth)

    # project points into 3D
    camera_params = cp.CameraParams(s.height_, s.width_, focal_length)
    point_cloud = camera_params.deproject(depth_im)
    orig_point_cloud = ip.PointCloudProcessing.remove_zero_points(point_cloud)

    # get round chessboard ind
    corner_px_round = np.round(corner_px).astype(np.uint16)
    corner_ind = ij_to_linear(corner_px_round[:,0], corner_px_round[:,1], s.width_)

    # fit a plane to the chessboard corners
    point_cloud_plane = point_cloud[:, corner_ind]
    n, mean_point_plane = ip.PointCloudProcessing.fit_plane(point_cloud_plane)

    # threshold to find objects on the table
    mean_point_plane = mean_point_plane + eps * n.reshape(3,1)
    _, points_uninterest = ip.PointCloudProcessing.prune_points_above_plane(point_cloud, n, mean_point_plane)
    pixels_uninterest = linear_to_ij(points_uninterest, s.width_)
    depth_im[pixels_uninterest[:,1], pixels_uninterest[:,0]] = 0.0

    # crop image at center
    depth_im_crop = ip.DepthImageProcessing.crop_center(depth_im, depth_im_crop_dim, depth_im_crop_dim)
    if debug:
        plt.figure()
        plt.imshow(depth_im_crop, cmap=plt.cm.Greys_r, interpolation='none')
        plt.title('Cropped raw depth image', fontsize=font_size)
        plt.show()

    # remove spurious points by finding the largest connected object
    binary_im = ip.DepthImageProcessing.depth_to_binary(depth_im_crop)
    binary_im_ch = ip.BinaryImageProcessing.prune_contours(binary_im)
    depth_im_crop = ip.DepthImageProcessing.mask_binary(depth_im_crop, binary_im)

    # filter
    depth_im_crop = skf.median_filter(depth_im_crop, size=depth_im_median_filter_dim)
    binary_mask = 1.0 * snm.binary_erosion(depth_im_crop, structure=np.ones((depth_im_erosion_filter_dim, depth_im_erosion_filter_dim)))
    depth_im_crop = ip.DepthImageProcessing.mask_binary(depth_im_crop, binary_mask)

    # center nonzero depth
    depth_im_crop_tf, diff_px = ip.DepthImageProcessing.center_nonzero_depth(depth_im_crop)
    if debug:
        plt.figure()
        plt.imshow(depth_im_crop_tf, cmap=plt.cm.Greys_r, interpolation='none')
        plt.title('Cropped, centered, and filtered depth image', fontsize=font_size)
        plt.show()

    # compute normals for registration
    camera_c_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], focal_length,
                                      cx=depth_im_crop.shape[0]/2.0, cy=depth_im_crop.shape[1]/2.0)

    query_object_normals, query_point_cloud = ip.DepthImageProcessing.compute_normals(depth_im_crop_tf, camera_c_params)

    # create CNN-sized depth image for indexing
    depth_im_crop_tf_cnn = ip.DepthImageProcessing.crop_center(depth_im_crop_tf, cnn_dim, cnn_dim)
    binary_im_crop_tf_cnn = ip.DepthImageProcessing.depth_to_binary(depth_im_crop_tf_cnn)
    grayscale_im_crop_tf_cnn = ip.BinaryImageProcessing.binary_to_grayscale(binary_im_crop_tf_cnn)
    
    # yes, we actually need to save the image to get the JPEG compression artifacts... otherwise the nets suck 
    grayscale_im_crop_tf_cnn = Image.fromarray(grayscale_im_crop_tf_cnn)
    grayscale_im_crop_tf_cnn.save(cache_image_filename, 'JPEG')
    grayscale_im_crop_tf_cnn = np.array(Image.open(cache_image_filename).convert('RGB'))

    if debug:
        plt.figure()
        plt.imshow(grayscale_im_crop_tf_cnn, cmap=plt.cm.Greys_r, interpolation='none')
        plt.title('Grayscale image for CNN', fontsize=font_size)
        plt.show()
    query_image = ri.RenderedImage(grayscale_im_crop_tf_cnn, np.zeros(3), np.zeros(3), np.zeros(3))

    # index the database for similar objects
    cnn_indexer = dbi.CNN_Hdf5ObjectIndexer(object_key, dataset, config)
    nearest_neighbors = cnn_indexer.k_nearest(query_image, k=num_nearest_neighbors)
    nearest_images = nearest_neighbors[0]
    nearest_distances = nearest_neighbors[1]
    
    # visualization for debugging
    if vis:
        plt.figure()
        plt.subplot(2, num_nearest_neighbors, math.ceil(float(num_nearest_neighbors)/2))
        plt.imshow(query_image.image, cmap=plt.cm.Greys_r, interpolation='none')
        plt.title('QUERY IMAGE', fontsize=font_size)
        plt.axis('off')

        for j, (image, distance) in enumerate(zip(nearest_images, nearest_distances)):
            plt.subplot(2, num_nearest_neighbors, j+num_nearest_neighbors+1)
            plt.imshow(image.image, cmap=plt.cm.Greys_r, interpolation='none')
            plt.title('NEIGHBOR %d, DISTANCE = %f' %(j, distance), fontsize=font_size)
            plt.axis('off')
        plt.show()

    # load object mesh
    obj = dataset.graspable(object_key)
    object_mesh = obj.mesh

    # register
    min_cost = np.inf
    best_reg = None
    best_index = -1
    for i, neighbor_image in enumerate(nearest_images):
        # get source object points
        source_mesh_tf = neighbor_image.camera_to_object_transform()        
        source_object_mesh = object_mesh.transform(source_mesh_tf)
        source_object_mesh.compute_normals()
        source_object_points = np.array(source_object_mesh.vertices())
        source_object_normals = np.array(source_object_mesh.normals())

        target_object_points = ip.PointCloudProcessing.remove_zero_points(query_point_cloud).T
        target_object_normals =ip.NormalCloudProcessing.remove_zero_normals(query_object_normals).T

        if debug:
            subsample_inds = np.arange(target_object_points.shape[0])[::20]
            mlab.figure()
            mlab.points3d(source_object_points[:,0], source_object_points[:,1], source_object_points[:,2], color=(1,0,0), scale_factor = 0.005)
            mlab.points3d(target_object_points[subsample_inds,0], target_object_points[subsample_inds,1], target_object_points[subsample_inds,2], color=(0, 1,0), scale_factor = 0.005)
            mlab.show()

        # point to plane ICP solver
        ppis = reg.PointToPlaneICPSolver(sample_size=icp_sample_size, gamma=icp_relative_point_plane_cost, mu=icp_regularizatiion_lambda)
        ppfm = fm.PointToPlaneFeatureMatcher(dist_thresh=feature_matcher_dist_thresh, norm_thresh=feature_matcher_norm_thresh) 
        registration = ppis.register(source_object_points, target_object_points, source_object_normals, target_object_normals, ppfm, num_iterations=num_registration_iters, vis=debug)

        logging.info('Neighbor %d registration cost %f' %(i, registration.cost))
        if registration.cost < min_cost:
            min_cost = registration.cost
            best_reg = registration
            best_tf_camera_camera_c = stf.SimilarityTransform3D(pose=tfx.pose(best_reg.R, best_reg.t))
            best_index = i

    # Great, now when you return... put the registration visualization code and test 
    best_tf_obj_camera = nearest_images[best_index].camera_to_object_transform()
    best_tf_obj_camera_c = best_tf_camera_camera_c.pose.matrix.dot(best_tf_obj_camera.pose.matrix)
    best_tf_obj_camera_c = stf.SimilarityTransform3D(pose=tfx.pose(best_tf_obj_camera_c))
    
    # project mesh at the transform into the image
    object_mesh_tf = object_mesh.transform(best_tf_obj_camera_c)
    object_point_cloud = np.array(object_mesh_tf.vertices()).T

    object_mesh_proj_pixels, mesh_valid_ind = camera_c_params.project(object_point_cloud)
    if debug:
        plt.figure()
        plt.imshow(depth_im_crop_tf, cmap=plt.cm.Greys_r, interpolation='none')
        plt.scatter(object_mesh_proj_pixels[0,mesh_valid_ind], object_mesh_proj_pixels[1,mesh_valid_ind], s=80, c='r')
        plt.title('Projected object mesh pixels', fontsize=font_size)
        plt.show()

    # find transform relative to the original image
    tf_camera_c_camera_p = ip.DepthImageProcessing.image_shift_to_transform(depth_im_crop_tf, depth_im_crop, camera_c_params, -diff_px)
    tf_obj_camera_p = tf_camera_c_camera_p.pose.matrix.dot(best_tf_obj_camera_c.pose.matrix)
    tf_obj_camera_p = stf.SimilarityTransform3D(pose=tfx.pose(tf_obj_camera_p))

    if vis:
        object_mesh_tf = object_mesh.transform(tf_obj_camera_p)
        source_object_points = np.array(object_mesh_tf.vertices())
        target_object_points = orig_point_cloud.T
        subsample_inds = np.arange(target_object_points.shape[0])[::20]

        mlab.figure()
        mlab.points3d(source_object_points[:,0], source_object_points[:,1], source_object_points[:,2], color=(1,0,0), scale_factor = 0.005)
        mlab.points3d(target_object_points[subsample_inds,0], target_object_points[subsample_inds,1], target_object_points[subsample_inds,2], color=(0, 1,0), scale_factor = 0.005)
        mlab.title('Object Mesh (Red) Overlayed \n on Point Cloud (Green)', size=font_size)
        mlab.show()

    IPython.embed()
