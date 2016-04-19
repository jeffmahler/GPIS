"""
Script to evaluate the probability of success for a few grasps on Zeke, logging the target states and the predicted quality in simulation
Authors: Jeff Mahler and Jacky Liang
"""
import copy
import csv
import logging
import IPython
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mayavi.mlab as mv
from random import choice
import os
import random
import shutil
import sys
sys.path.append("src/grasp_selection/control/DexControls")
import time

from DexAngles import DexAngles
from DexConstants import DexConstants
from DexController import DexController
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from ZekeState import ZekeState

import camera_params as cp
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import mab_single_object_objective as msoo
import mayavi_visualizer as mvis
import obj_file
import quality
import rgbd_sensor as rs
import similarity_tf as stf
import tabletop_object_registration as tor
import termination_conditions as tc
import tfx

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

# Grasp execution main function
def test_registration(graspable, stable_pose, dataset, registration_solver, camera, config):
    debug = config['debug']
    load_object = config['load_object']
    alpha = config['alpha']
    center_scale = config['center_scale']
    tube_radius = config['tube_radius']
    table_extent = config['table_extent']
    lift_height = config['lift_height']
    num_grasp_views = config['num_grasp_views']
    cam_dist = config['cam_dist']

    logging_dir = config['experiment_dir']        
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    registration_solver.log_to(logging_dir)

    # retrieve object pose from camera
    logging.info('Registering object')
    depth_im = camera.get_depth_image()
    color_im = camera.get_color_image()
                
    # get point cloud (for debugging only)
    camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], fx=config['registration']['focal_length'])
    points_3d = camera_params.deproject(depth_im)
    subsample_inds = np.arange(points_3d.shape[1])[::10]
    points_3d = points_3d[:,subsample_inds]

    # register
    reg_result = registration_solver.register(copy.copy(color_im), copy.copy(depth_im), debug=debug)
    T_camera_obj = reg_result.tf_camera_obj
    T_camera_obj.from_frame = 'obj'
    T_camera_obj.to_frame = 'camera'
    T_obj_camera = T_camera_obj.inverse()    

    # save depth and color images
    min_d = np.min(depth_im)
    max_d = np.max(depth_im)
    depth_im2 = 255.0 * (depth_im - min_d) / (max_d - min_d)
    depth_im2 = Image.fromarray(depth_im2.astype(np.uint8))
    filename = 'depth.png'
    depth_im2.save(os.path.join(logging_dir, filename))
    color_im2 = Image.fromarray(color_im)
    filename = 'color.png'
    color_im2.save(os.path.join(logging_dir, filename))

    # transform the mesh to the stable pose to get a z offset from the table
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
    object_mesh = graspable.mesh
    object_mesh_tf = object_mesh.transform(T_obj_stp)
    mn, mx = object_mesh_tf.bounding_box()
    z = mn[2]
    
    # define poses of camera, table, object, tec
    T_world = stf.SimilarityTransform3D(from_frame='world', to_frame='world')
    R_table_world = np.eye(3)
    T_table_world = stf.SimilarityTransform3D(pose=tfx.pose(R_table_world, np.zeros(3)), from_frame='world', to_frame='table')
            
    R_camera_table = np.load('data/calibration/rotation_camera_cb.npy')
    t_camera_table = np.load('data/calibration/translation_camera_cb.npy')
    cb_points_camera = np.load('data/calibration/corners_cb.npy')
    T_camera_table = stf.SimilarityTransform3D(tfx.pose(R_camera_table, t_camera_table), from_frame='table', to_frame='camera')
    T_camera_world = T_camera_table.dot(T_table_world)
    T_world_camera = T_camera_world.inverse()
    
    R_stp_obj = stable_pose.r
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')
            
    t_stp_table = np.array([0, 0, z])
    T_stp_table = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), t_stp_table), from_frame='table', to_frame='stp')
    
    T_obj_world = T_obj_camera.dot(T_camera_world)
            
    # visualize the robot's understanding of the world
    logging.info('Displaying robot world state')
    mv.clf()
    mvis.MayaviVisualizer.plot_table(T_table_world, d=table_extent)
    mvis.MayaviVisualizer.plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
    mvis.MayaviVisualizer.plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
    mvis.MayaviVisualizer.plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
    mvis.MayaviVisualizer.plot_mesh(object_mesh, T_obj_world)
    mvis.MayaviVisualizer.plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
    mvis.MayaviVisualizer.plot_point_cloud(points_3d, T_world_camera, color=(0,1,0), scale=0.0025)
    mv.view(focalpoint=(0,0,0))
    mv.show()
            
if __name__ == '__main__':
    random.seed(102)
    np.random.seed(102)

    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    output_dir = sys.argv[2]

    # open config and read params
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    object_name = config['object_name']
    num_grasp_views = config['num_grasp_views']
    max_iter = config['max_iter']
    snapshot_rate = config['snapshot_rate']

    # open database and dataset
    database = db.Hdf5Database(database_filename, config)#, access_level=db.READ_WRITE_ACCESS)
    ds = database.dataset(dataset_name)

    # setup output directories and logging (TODO: make experiment wrapper class in future)
    experiment_id = 'registration_test_%s' %(gen_experiment_id())
    experiment_dir = os.path.join(output_dir, experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    config['experiment_dir'] = experiment_dir
    experiment_log = os.path.join(experiment_dir, experiment_id +'.log')
    hdlr = logging.FileHandler(experiment_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr) 

    # copy over config file
    config_path, config_fileroot = os.path.split(config_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_fileroot))
    logging.info('RUNNING REGISTRATION TEST %s' %(experiment_id))

    # read the grasp metrics and features
    graspable = ds.graspable(object_name)
    stable_poses = ds.stable_poses(object_name)

    """
    filename = '/mnt/terastation/shape_data/dexnet_physical_experiments/pipe_connector_dec.obj'
    of = obj_file.ObjFile(filename)
    object_mesh = of.read()

    #object_mesh, _ = graspable.mesh.subdivide(min_triangle_length=config['min_triangle_length'])
    ds.update_mesh(object_name, object_mesh)
    database.close()
    exit(0)
    """

    T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')
    for stable_pose in stable_poses:
        print 'Stable pose', stable_pose.id
        mv.figure()
        mvis.MayaviVisualizer.plot_stable_pose(graspable.mesh, stable_pose, T_table_world)
        mv.show()
    exit(0)

    stable_pose = stable_poses[config['stable_pose_index']]
    
    # HACK to fix a y-axis orientation bug in the stable pose code
    if np.abs(np.linalg.det(stable_pose.r) + 1) < 0.01:
        stable_pose.r[1,:] = -stable_pose.r[1,:]

    # preload registration solver
    registration_solver = tor.KnownObjectStablePoseTabletopRegistrationSolver(object_name, stable_pose.id, ds, config)

    # init hardware
    logging.info('Initializing camera')
    camera = rs.RgbdSensor()

    # run registration
    test_registration(graspable, stable_pose, ds, registration_solver, camera, config)
