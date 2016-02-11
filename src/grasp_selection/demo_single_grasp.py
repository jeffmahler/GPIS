"""
Script to demo a single grasp of the spray bottle on Zeke
Authors: Jeff Mahler and Jacky Liang
"""
import logging
import IPython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mayavi.mlab as mv
from random import choice
import os
import sys
sys.path.append("src/grasp_selection/control/DexControls")
import time

from DexController import DexController
from ZekeState import ZekeState
from IzzyState import IzzyState

import database as db
import experiment_config as ec
import rgbd_sensor as rs
import similarity_tf as stf
import tabletop_object_registration as tor
import tfx

# MAYAVI VISUALIZER
def mv_plot_table(T_table_world, d=0.5):
    """ Plots a table in pose T """
    table_vertices = np.array([[d, d, 0],
                               [d, -d, 0],
                               [-d, d, 0],
                               [-d, -d, 0]])
    table_vertices_tf = T_table_world.apply(table_vertices.T).T
    table_tris = np.array([[0, 1, 2], [1, 2, 3]])
    mv.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2], table_tris, representation='surface', color=(0,0,0))

def mv_plot_pose(T_frame_world, alpha=0.5, tube_radius=0.005, center_scale=0.025):
    T_world_frame = T_frame_world.inverse()
    R = T_world_frame.rotation
    t = T_world_frame.translation

    x_axis_tf = np.array([t, t + alpha * R[:,0]])
    y_axis_tf = np.array([t, t + alpha * R[:,1]])
    z_axis_tf = np.array([t, t + alpha * R[:,2]])
        
    mv.points3d(t[0], t[1], t[2], color=(1,1,1), scale_factor=center_scale)
    
    mv.plot3d(x_axis_tf[:,0], x_axis_tf[:,1], x_axis_tf[:,2], color=(1,0,0), tube_radius=tube_radius)
    mv.plot3d(y_axis_tf[:,0], y_axis_tf[:,1], y_axis_tf[:,2], color=(0,1,0), tube_radius=tube_radius)
    mv.plot3d(z_axis_tf[:,0], z_axis_tf[:,1], z_axis_tf[:,2], color=(0,0,1), tube_radius=tube_radius)

    mv.text3d(t[0], t[1], t[2], ' %s' %T_frame_world.to_frame.upper(), scale=0.01)

def mv_plot_mesh(mesh, T_mesh_world, style='wireframe', color=(0.5,0.5,0.5)):
    mesh_tf = mesh.transform(T_mesh_world.inverse())
    mesh_tf.visualize(style=style, color=color)

def mv_plot_point_cloud(point_cloud, T_points_world, color=(0,1,0), scale=0.01):
    point_cloud_tf = T_points_world.apply(point_cloud).T
    mv.points3d(point_cloud_tf[:,0], point_cloud_tf[:,1], point_cloud_tf[:,2], color=color, scale_factor=scale)

# Grasp execution main function
def test_grasp_execution(graspable, grasp, stable_pose, dataset, config, debug=False):
    load_object = False

    # check collisions
    grasp = grasp.grasp_aligned_with_stable_pose(stable_pose)
    debug_output = []
    does_collide = grasp.collides_with_stable_pose(stable_pose, debug_output)
    collision_box_vertices = np.array(debug_output[0]).T
    if does_collide:
        logging.error('Grasp is in collision')
        return

    # init hardware
    ctrl = DexController()
    camera = rs.RgbdSensor()
    
    """
    extension = 0.1039
    center_angle = 3.4611
    edge_angle = 3.8372

    ctrl.reset_object()
    while not ctrl._izzy.is_action_complete():
        time.sleep(0.01)
    
    ctrl._izzy.gotoState(ZekeState([center_angle, None, extension, None, None, None]))
    while not ctrl._izzy.is_action_complete():
        time.sleep(0.01)
        
    state = ctrl._izzy.getState()
    print 'Center 1', state
    time.sleep(5)

    ctrl._izzy.gotoState(ZekeState([edge_angle, None, extension, None, None, None]))
    while not ctrl._izzy.is_action_complete():
        time.sleep(0.01)

    state = ctrl._izzy.getState()
    print 'Ext 1', state
    time.sleep(5)

    ctrl._izzy.gotoState(ZekeState([center_angle, None, extension, None, None, None]))
    while not ctrl._izzy.is_action_complete():
        time.sleep(0.01)
        
    state = ctrl._izzy.getState()
    print 'Center 2', state
    time.sleep(5)

    ctrl._izzy.gotoState(ZekeState([edge_angle, None, extension, None, None, None]))
    while not ctrl._izzy.is_action_complete():
        time.sleep(0.01)

    state = ctrl._izzy.getState()
    print 'Ext 2', state

    IPython.embed()
    ctrl.stop()
    exit(0)
    """

    if True:
    #try:
        if not load_object:
            # move the arm out of the way
            ctrl.reset_object()
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)

            """
            time.sleep(5)
            ctrl._izzy.gotoState(ZekeState([3.3862, None, 0.09, None, None, None]))

            time.sleep(10)
            ctrl.stop()
            exit(0)
            """

            # prompt for arm
            yesno = raw_input('Place object. Hit [ENTER] when done')

            # tabletop chessboard registration
            depth_im = camera.get_depth_image()
            color_im = camera.get_color_image()
            registration_solver = tor.KnownObjectTabletopRegistrationSolver()
            reg_result = registration_solver.register(color_im, depth_im, graspable.key, dataset, config, debug=debug)
            T_camera_obj = reg_result.tf_camera_obj
            T_camera_obj.from_frame = 'obj'
            T_camera_obj.to_frame = 'camera'
            T_obj_camera = T_camera_obj.inverse()    
    
        else:
            T_camera_obj = stf.SimilarityTransform3D()
            T_camera_obj.load('data/calibration/spray_pose.stf')
            T_camera_obj.from_frame = 'obj'
            T_camera_obj.to_frame = 'camera'
            T_obj_camera = T_camera_obj.inverse()

        # transform the mesh
        T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
        object_mesh = graspable.mesh
        object_mesh_tf = object_mesh.transform(T_obj_stp)
        mn, mx = object_mesh_tf.bounding_box()
        z = mn[2]
    
        # define poses
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
        
        T_gripper_obj = grasp.gripper_transform(gripper='zeke')
    
        T_gripper_world = T_gripper_obj.dot(T_obj_world)

        mv.figure()
        mv_plot_table(T_table_world, d=table_extent)
        mv_plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        #mv_plot_pose(T_table_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        #mv_plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_mesh(object_mesh, T_obj_world)
        #mv_plot_point_cloud(collision_box_vertices, T_obj_world)
        mv_plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
        mv.show()

        # execute the grasp
        ctrl.do_grasp(T_gripper_world.inverse())
        ctrl.plot_approach_angle()
        while not ctrl._izzy.is_action_complete():
            time.sleep(0.01)
        grasp_state, _ = ctrl.getState()
        high_state = grasp_state.copy().set_arm_elev(0.25)
        high_state.set_gripper_grip(IzzyState.MIN_STATE().gripper_grip)
        print grasp_state
        print "Sending high low states..."
        ctrl._izzy.gotoState(high_state)
        while not ctrl._izzy.is_action_complete():
            time.sleep(0.01)
        ctrl._izzy.plot()
        ctrl.stop()
    #except Exception as e:
    #    ctrl.stop()
    #    raise e

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    config_file = "cfg/demo/demo_single_grasp.yaml"
    database_filename = "/mnt/terastation/shape_data/MASTER_DB_v3/aselab_db.hdf5"
    dataset_name = "aselab"
    item_name = "spray"

    alpha = 0.05
    center_scale = 0.0075
    tube_radius = 0.0025
    table_extent = 0.5

    config = ec.ExperimentConfig(config_file)
    database = db.Hdf5Database(database_filename, config)

    # read the grasp metrics and features
    ds = database.dataset(dataset_name)

    graspable = ds.graspable(item_name)
    grasps = ds.grasps(item_name)
    grasp_features = ds.grasp_features(item_name, grasps)
    grasp_metrics = ds.grasp_metrics(item_name, grasps)
    stable_poses = ds.stable_poses(item_name)
    
    metric = 'pfc_f_0.200000_tg_0.020000_rg_0.020000_to_0.020000_ro_0.020000'
    sorted_grasps, sorted_metrics = ds.sorted_grasps(item_name, metric)
    best_grasp = sorted_grasps[100]

    p = len(stable_poses)
    n = len(grasps)

    test_grasp_execution(graspable, best_grasp, stable_poses[2], ds, config)
