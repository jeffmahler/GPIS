from primesense import openni2
import caffe
import glob
import logging
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
import experiment_config as ec
import feature_file as ff
import feature_matcher as fm
import mesh
import obj_file as objf
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
        depth_image = depth_image * 0.001
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
    load = False
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
        num_images = 1
        depth_im = np.zeros([s.height_, s.width_])
        counts = np.zeros([s.height_, s.width_])
        for i in range(num_images):
            new_depth_im = s.get_depth_image()

            depth_im = depth_im + new_depth_im
            counts = counts + np.array(new_depth_im > 0.0)

        depth_im[depth_im > 0] = depth_im[depth_im > 0] / counts[depth_im > 0]
        #f = open('data/test/rgbd/depth_im.npy', 'w')
        #np.save(f, depth_im)

        color_im = s.get_color_image()
        #f = open('data/test/rgbd/color_im.npy', 'w')
        #np.save(f, color_im)

        # find the chessboard
        corner_px = find_chessboard(color_im)
        #f = open('data/test/rgbd/corners.npy', 'w')
        #np.save(f, corner_px)

    depth_im[depth_im > 1.0] = 0.0

    # project points into 3D
    camera_params = cp.CameraParams(s.height_, s.width_, 525.)
    points_3d = camera_params.deproject(depth_im)

    # get round chessboard ind
    corner_px_round = np.round(corner_px).astype(np.uint16)
    corner_ind = ij_to_linear(corner_px_round[:,0], corner_px_round[:,1], s.width_)

    # fit a plane to the chessboard corners
    points_3d_plane = points_3d[:, corner_ind]
    X = np.c_[points_3d_plane[:2,:].T, np.ones(points_3d_plane.shape[1])]
    y = points_3d_plane[2,:].T
    A = X.T.dot(X)
    b = X.T.dot(y)
    w = np.linalg.inv(A).dot(b)
    n = np.array([w[0], w[1], -1])
    n = n / np.linalg.norm(n)
    mean_point_plane = np.mean(points_3d_plane, axis=1)
    mean_point_plane = np.reshape(mean_point_plane, [3, 1])

    # threshold to find objects on the table
    eps = 0.01
    mean_point_plane = mean_point_plane + eps * n.reshape(3,1)

    points_of_interest = (points_3d - np.tile(mean_point_plane, [1, points_3d.shape[1]])).T.dot(n) > 0
    points_of_interest = (points_3d[2,:] > 0) & points_of_interest
    points_of_interest = np.where(points_of_interest)[0]

    points_uninterest = np.setdiff1d(np.arange(s.width_ * s.height_), points_of_interest)
    pixels_uninterest = linear_to_ij(points_uninterest, s.width_)
    depth_im[pixels_uninterest[:,1], pixels_uninterest[:,0]] = 0.0

    # crop
    dim = 256
    start_row = 100
    start_col = 150
    depth_im_crop = depth_im[start_row:start_row+dim, start_col:start_col+dim]

    # remove spurious points by finding the largest connected object
    binary_im = 1 * (depth_im_crop > 0.0)
    binary_im = binary_im.astype(np.uint8)
    contours = cv2.findContours(binary_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # prune tiny connected components
    area_thresh = 200.0
    num_contours = len(contours[0])
    pruned_contours = []
    for i in range(num_contours):
        area = cv2.contourArea(contours[0][i])
        if area > area_thresh:
            pruned_contours.append(contours[0][i])

    # mask out bad areas in the image
    binary_im_ch = np.zeros([binary_im.shape[0], binary_im.shape[1], 3])
    for contour in pruned_contours:
        cv2.fillPoly(binary_im_ch, pts=[contour], color=(255,255,255))
    ind = np.where(binary_im_ch == 0)
    depth_im_crop[ind[0], ind[1]] = 0.0

    # filter
    depth_im_crop = skf.median_filter(depth_im_crop, size=9.0)
    binary_mask = snm.binary_erosion(depth_im_crop, structure=np.ones((3,3)))
    ind = np.where(binary_mask == False)
    depth_im_crop[ind[0], ind[1]] = 0.0

    # center
    nonzero_px = np.where(depth_im_crop!=0)
    nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
    mean_px = np.mean(nonzero_px, axis=0)
    center_px = np.array(depth_im_crop.shape) / 2.0
    diff_px = center_px - mean_px
    nonzero_px_tf = nonzero_px + diff_px
    nonzero_px = nonzero_px.astype(np.uint16)
    nonzero_px_tf = nonzero_px_tf.astype(np.uint16)
    depth_im_crop_tf = np.zeros(depth_im_crop.shape)
    depth_im_crop_tf[nonzero_px_tf[:,0], nonzero_px_tf[:,1]] = depth_im_crop[nonzero_px[:,0], nonzero_px[:,1]]
    
    # reproject
    camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], 525.,
                                    cx=depth_im_crop.shape[0]/2.0, cy=depth_im_crop.shape[1]/2.0)
    points_3d = camera_params.deproject(depth_im_crop_tf)
    points_3d_grid = points_3d.T.reshape(depth_im_crop.shape[0], depth_im_crop.shape[1], 3)
    points_of_interest = np.where(points_3d[2,:] != 0.0)[0]
    points_3d = points_3d[:, points_of_interest]

    # compute normals
    normals = np.zeros([depth_im_crop.shape[0], depth_im_crop.shape[1], 3])
    for i in range(depth_im_crop.shape[0]-1):
        for j in range(depth_im_crop.shape[1]-1):
            p = points_3d_grid[i,j,:]
            p_r = points_3d_grid[i,j+1,:]
            p_b = points_3d_grid[i+1,j,:]
            if np.linalg.norm(p) > 0 and np.linalg.norm(p_r) > 0 and np.linalg.norm(p_b) > 0:
                v_r = p_r - p
                v_r = v_r / np.linalg.norm(v_r)
                v_b = p_b - p
                v_b = v_b / np.linalg.norm(v_b)
                normals[i,j,:] = np.cross(v_b, v_r)
                normals[i,j,:] = normals[i,j,:] / np.linalg.norm(normals[i,j,:])
    normals = normals.reshape(depth_im_crop.shape[0]*depth_im_crop.shape[1], 3).T
    normals = normals[:, points_of_interest]

    # extract shot features
    """
    partial_mesh_filename = 'partial_mesh.obj'
    partial_shot_filename = 'partial_mesh.ftr'
    partial_mesh = mesh.Mesh3D(points_3d.T.tolist(), [])
    of = objf.ObjFile(partial_mesh_filename)
    of.write(partial_mesh)
    shot_os_call = 'bin/shot_extractor %s %s' %(partial_mesh_filename, partial_shot_filename)
    print 'Calling', shot_os_call
    os.system(shot_os_call)

    # load shot features
    partial_feat_file = ff.LocalFeatureFile(partial_shot_filename)
    partial_features = partial_feat_file.read()
    """

    # load spray bottle mesh
    mesh_filename = '/mnt/terastation/shape_data/aselab/spray_dec.obj'
    shot_filename = '/mnt/terastation/shape_data/aselab/spray_dec.ftr'
    of = objf.ObjFile(mesh_filename)
    m = of.read()
    vertices = np.array(m.vertices()).T
    
    """
    feat_file = ff.LocalFeatureFile(shot_filename)
    target_features = feat_file.read()

    # match
    feat_matcher = fm.RawDistanceFeatureMatcher()
    corrs = feat_matcher.match(partial_features, target_features)
    """
    """
    points_3d = points_3d[:, np.arange(points_3d.shape[1], step=50)]
    ax = plt.gca(projection = '3d')
    ax.scatter(points_3d[0,:], points_3d[1,:], points_3d[2,:], c='b')
    ax.scatter(vertices[0,:], vertices[1,:], vertices[2,:], c='g')

    for i in range(corrs.source_points.shape[0]):
        ax.scatter(corrs.source_points[i,0], corrs.source_points[i,1], corrs.source_points[i,2], s=120, c=u'r')
        ax.scatter(corrs.target_points[i,0], corrs.target_points[i,1], corrs.target_points[i,2], s=120, c=u'm')
    """

    """
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    ax.set_zlim3d(0, 2.0)
    """

    """
    num_plot = 10
    for i in range(num_plot):
        t = i * (1.0 / num_plot)
        ax.scatter(t * n[0], t * n[1], t * n[2], c=u'g', s=150)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    ax.set_zlim3d(0, 2.0)
    """

    stp_filename = '/mnt/terastation/shape_data/aselab/spray.stp'
    sf = stp_file.StablePoseFile()
    stps = sf.read(stp_filename)

    # load the templates
    template_dir = '/home/jmahler/jeff_working/GPIS/data/spray_templates'
    template_filenames = []
    object_poses = []
    thetas = []
    phis = []
    rots = []
    ts = []
    poses = []
    stpss = []
    camera_pose_arrs = []
    table_normals = [] 
    for i, d in enumerate(os.listdir(template_dir)):
        subdir = os.path.join(template_dir, d)
        stp = stps[2]

        camera_pose_arr = np.genfromtxt(os.path.join(subdir, 'camera_table.csv'), delimiter=',', dtype=np.dtype(str))
        for j in range(camera_pose_arr.shape[0]):
            camera_xyz_w = camera_pose_arr[j, :3].astype(np.float)
            camera_rot_w = camera_pose_arr[j, 3:6].astype(np.float)
            camera_int_pt_w = camera_pose_arr[j, 6:9].astype(np.float)
            camera_xyz_obj_p = camera_xyz_w - camera_int_pt_w
            camera_dist_xy = np.linalg.norm(camera_xyz_w[:2])
            z = [0,0,np.linalg.norm(camera_xyz_w[:3])]

            theta = camera_rot_w[0] * np.pi / 180.0# + np.pi / 2
            phi = -camera_rot_w[2] * np.pi / 180.0 + np.pi / 2.0

            camera_rot_obj_p_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                                           [np.sin(phi), np.cos(phi), 0],
                                           [0, 0, 1]])

            camera_rot_obj_p_x = np.array([[1, 0, 0],
                                           [0, np.cos(theta), -np.sin(theta)],
                                           [0, np.sin(theta), np.cos(theta)]])

            #camera_rot_obj_p_x = np.eye(3)
            #camera_rot_obj_p_z = np.eye(3)

            camera_md = np.array([[0, 1, 0],[1, 0, 0],[0,0,-1]])
            #camera_md = np.eye(3)
            camera_rot_obj_p = camera_md.dot(camera_rot_obj_p_z.dot(camera_rot_obj_p_x))
            camera_rot_obj_p = camera_rot_obj_p.T

            """
            canon_axes = np.eye(3)
            axes_tf = camera_rot_obj_p.dot(canon_axes)

            if subdir.find('2') != -1 and camera_pose_arr[j,-1].find('0_3_3') != -1:
                print 'T', camera_rot_w[0]
                print 'P', camera_rot_w[2]
                mlab.figure()
                mlab.points3d(canon_axes[:,0], canon_axes[:,1], canon_axes[:,2], scale_factor=0.2, color=(1,0,0))
                mlab.points3d(axes_tf[0,2], axes_tf[1,2], axes_tf[2,2], scale_factor=0.2, color=(0,1,0))
                mlab.points3d(axes_tf[0,1], axes_tf[1,1], axes_tf[2,1], scale_factor=0.2, color=(0,0,1))
                mlab.points3d(axes_tf[0,0], axes_tf[1,0], axes_tf[2,0], scale_factor=0.2, color=(0,1,1))
                mlab.points3d(0,0,0,scale_factor=0.1,color=(1,0,0))
                mlab.axes()
                mlab.show()
            #camera_rot_obj_p = np.eye(3)
            #a = np.eye(3)
            #a[0,0] = -1
            #a[2,2] = -1
            """

            T_obj_obj_p = tfx.pose(stp.r).matrix
            T_obj_p_camera = tfx.pose(camera_rot_obj_p, z).matrix#, camera_rot_obj_p.dot(camera_xyz_obj_p)).matrix
            #T_obj_p_camera = tfx.pose(camera_rot_obj_p).matrix
            T_obj_camera = T_obj_p_camera.dot(T_obj_obj_p)

            # rotate to match table normal for 2D search
            stp_table_basis = np.array(T_obj_camera[:3,:3].dot(stp.r.T))
            table_tan_x = np.array([-n[1], n[0], 0])
            table_tan_x = table_tan_x / np.linalg.norm(table_tan_x)
            table_tan_y = np.cross(n, table_tan_x)
            t0 = stp_table_basis[:,0].dot(table_tan_x)
            t1 = stp_table_basis[:,0].dot(table_tan_y)
            xp = t0*table_tan_x + t1*table_tan_y
            xp = xp / np.linalg.norm(xp)
            yp = np.cross(n, xp)
            Ap = np.c_[xp, yp, n]
            Rp = Ap.dot(stp_table_basis.T)
            T_obj_p_camera = tfx.pose(Rp.dot(camera_rot_obj_p), z).matrix#, camera_rot_obj_p.dot(camera_xyz_obj_p)).matrix
            T_obj_camera = T_obj_p_camera.dot(T_obj_obj_p)
            stp_table_basis = np.array(T_obj_camera[:3,:3].dot(stp.r.T))

            stpss.append(stp)
            rots.append(camera_rot_obj_p)
            ts.append(camera_xyz_obj_p)
            poses.append(T_obj_p_camera)
            table_normals.append(stp_table_basis)

            thetas.append(theta)
            phis.append(phi)
            camera_pose_arrs.append(camera_pose_arr[j,:])
            object_poses.append(stf.SimilarityTransform3D(pose=tfx.pose(T_obj_camera)))
            template_filenames.append(os.path.join(subdir,camera_pose_arr[j,-1])+'.jpg')

        #template_filenames.extend(glob.glob('%s/*segmask*.jpg' %(subdir)))

    template_images = []
    for f in template_filenames:
        template_images.append(np.array(Image.open(f)))

    binary_im_crop_tf = 255 * (depth_im_crop_tf > 0)
    binary_im_crop_tf = binary_im_crop_tf.astype(np.uint8)
    binary_im_filename = '/home/jmahler/jeff_working/GPIS/data/spray_binary.jpg'
    bict = Image.fromarray(binary_im_crop_tf)
    bict.save(binary_im_filename)

    dists = []
    for im in template_images:
        dists.append(np.linalg.norm((binary_im_crop_tf - im[:,:,0]).squeeze()))
    
    config = ec.ExperimentConfig('/home/jmahler/jeff_working/GPIS/cfg/test_registration.yaml')
    mv = MVCNNBatchFeatureExtractor(config)
    template_vecs = mv._forward_pass(template_filenames)
    target_vec = mv._forward_pass([binary_im_filename])

    template_vecs = template_vecs.reshape([template_vecs.shape[0], -1])
    target_vec = target_vec.reshape([target_vec.shape[0], -1])

    target_vecs = np.tile(target_vec, [template_vecs.shape[0], 1])
    dists = np.linalg.norm(template_vecs - target_vecs, axis=1)
    dists_and_indices = zip(dists, range(len(dists)))
    dists_and_indices.sort(key=lambda x: x[0])

    # plot everything
    min_cost = np.inf
    best_reg = None
    best_index = -1
    for i in range(0,3):
        index = dists_and_indices[i][1]
        best_tf = object_poses[index]
        theta = thetas[index]
        phi = phis[index]
        camera_arr = camera_pose_arrs[index]
        r = rots[index]
        t = ts[index]
        p = poses[index]
        s = stpss[index]
        stp_n = table_normals[index]

        print 'Theta', theta * 180 / np.pi
        print 'Phi', phi * 180 / np.pi
        print 'N', stp_n
        print index
        print template_filenames[index]

        m_tf = m.transform(best_tf)
        m_tf.compute_normals()
        m_normals = np.array(m_tf.normals())

        """
        plt.figure()
        plt.imshow(template_images[index], cmap=plt.cm.Greys_r, interpolation='none')
        plt.show()
        
        mlab.figure()
        vertex_array = np.array(m_tf.vertices())
        #mlab.points3d(vertex_array[:,0], vertex_array[:,1], vertex_array[:,2], scale_factor=0.005, color=(0,1,0))
        mlab.points3d(points_3d[0,:], points_3d[1,:], points_3d[2,:], scale_factor=0.005, color=(1,0,0))
        m_tf.visualize()

        canon_axes = 0.1*np.eye(3)
        axes_tf = best_tf.rotation.dot(canon_axes)
        mlab.points3d(0,0,0,scale_factor=0.01,color=(1,0,0))
        mlab.points3d(axes_tf[0,2], axes_tf[1,2], axes_tf[2,2], scale_factor=0.02, color=(0,1,0))
        mlab.points3d(axes_tf[0,1], axes_tf[1,1], axes_tf[2,1], scale_factor=0.02, color=(0,0,1))
        mlab.points3d(axes_tf[0,0], axes_tf[1,0], axes_tf[2,0], scale_factor=0.02, color=(0,1,1))
        mlab.points3d(0.1*n[0], 0.1*n[1], 0.1*n[2], scale_factor=0.02, color=(1,1,0))
        stp_n = stp_n[:,2]
        mlab.points3d(0.1*stp_n[0], 0.1*stp_n[1], 0.1*stp_n[2], scale_factor=0.02, color=(1,1,1))

        mlab.points3d(canon_axes[:,0], canon_axes[:,1], canon_axes[:,2], scale_factor=0.02, color=(1,0,0))
        canon_axes = 1.1 * canon_axes
        mlab.text3d(canon_axes[0,0], canon_axes[0,1], canon_axes[0,2], 'X', scale=0.01)
        mlab.text3d(canon_axes[1,0], canon_axes[1,1], canon_axes[1,2], 'Y', scale=0.01)
        mlab.text3d(canon_axes[2,0], canon_axes[2,1], canon_axes[2,2], 'Z', scale=0.01)
        #mlab.axes()
        mlab.show()
        """

        registration = reg.point_plane_icp_mesh_point_cloud(m_tf, points_3d, m_normals, normals, 10)
        logging.info('Cost %f' %(registration.cost))
        if registration.cost < min_cost:
            min_cost = registration.cost
            best_reg = registration
            best_index = index

    T_obj_camera = object_poses[best_index]
    T_obj_camera_p = stf.SimilarityTransform3D(pose=tfx.pose(best_reg.R, best_reg.t))
    M = T_obj_camera_p.pose.matrix.dot(T_obj_camera.pose.matrix)
    T_obj_camera = stf.SimilarityTransform3D(pose=tfx.pose(M))

    m_tf = m.transform(T_obj_camera)
    mesh_points_3d = np.array(m_tf.vertices())

    mesh_proj_pixels, mesh_valid = camera_params.project(mesh_points_3d.T)
    mesh_valid_ind = np.where(mesh_valid)[0]
    plt.figure()
    plt.imshow(depth_im_crop_tf, cmap=plt.cm.Greys_r, interpolation='none')
    plt.scatter(mesh_proj_pixels[0,mesh_valid_ind], mesh_proj_pixels[1,mesh_valid_ind], s=80, c='r')

    # project pixels on the original image
    camera_params2 = cp.CameraParams(480, 640, 525)
    d_center = depth_im_crop_tf[center_px[0], center_px[1]]
    center_orig_im = np.array([start_row + mean_px[0], start_col + mean_px[1]])
    p_center = d_center * np.linalg.inv(camera_params2.K_).dot(np.array([center_orig_im[1], center_orig_im[0], 1]))
    t_c_cp = p_center
    t_c_cp[2] = 0
    T_c_cp = tfx.pose(np.eye(3), t_c_cp)
    T_obj_cp = T_c_cp.matrix.dot(T_obj_camera.pose.matrix)
    T_obj_cp = stf.SimilarityTransform3D(pose=tfx.pose(T_obj_cp))

    m_tf = m.transform(T_obj_cp)
    mesh_points_3d = np.array(m_tf.vertices())

    mesh_proj_pixels, mesh_valid = camera_params2.project(mesh_points_3d.T)
    mesh_valid_ind = np.where(mesh_valid)[0]
    plt.figure()
    plt.imshow(depth_im, cmap=plt.cm.Greys_r, interpolation='none')
    plt.scatter(mesh_proj_pixels[0,mesh_valid_ind], mesh_proj_pixels[1,mesh_valid_ind], s=80, c='r')
    plt.show()                                       

    IPython.embed()
    exit(0)

    mlab.figure()        
    mlab.points3d(points_3d[0,:], points_3d[1,:], points_3d[2,:], scale_factor=0.005, color=(1,0,0))
    mlab.points3d(mesh_points_3d[:,0], mesh_points_3d[:,1], mesh_points_3d[:,2], scale_factor=0.005, color=(0,1,0))
    mlab.axes()
    mlab.show()

    IPython.embed()
    exit(0)

    # display images
    for i in range(5):
        plt.figure()
        plt.imshow(template_images[dists_and_indices[i][1]])

    plt.figure()
    plt.imshow(depth_im_crop_tf, cmap=plt.cm.Greys_r, interpolation='none')

    plt.figure()
    plt.imshow(color_im)

    plt.show()

    IPython.embed()
    exit(0)
    
