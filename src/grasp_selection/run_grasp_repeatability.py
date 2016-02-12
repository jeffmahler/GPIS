"""
Script to evaluate the probability of success for a few grasps on Izzy, logging the target states and the predicted quality in simulation
Authors: Jeff Mahler and Jacky Liang
"""
import csv
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

from DexConstants import DexConstants
from DexController import DexController
from DexRobotIzzy import DexRobotIzzy
from ZekeState import ZekeState
from IzzyState import IzzyState

import database as db
import experiment_config as ec
import quality
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

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def compute_grasp_set(dataset, object_name, stable_pose, num_grasps, metric='pfc_f_0.200000_tg_0.020000_rg_0.020000_to_0.020000_ro_0.020000'):
    """ Add the best grasp according to PFC as well as num_grasps-1 uniformly at random from the remaining set """
    grasp_set = []

    # get sorted list of grasps to ensure that we get the top grasp
    sorted_grasps, sorted_metrics = dataset.sorted_grasps(object_name, metric)
    num_total_grasps = len(sorted_grasps)
    best_grasp = sorted_grasps[0]
    grasp_set.append(best_grasp)

    # get random indices
    indices = np.arange(1, num_total_grasps)
    np.random.shuffle(indices)

    i = 0
    while len(grasp_set) < num_grasps:
        grasp_candidate = sorted_grasps[indices[i]]
        grasp_candidate = grasp_candidate.grasp_aligned_with_stable_pose(stable_pose)
        in_collision = grasp_candidate.collides_with_stable_pose(stable_pose)
        center_rel_table = grasp_candidate.center - stable_pose.x0
        dist_to_table = center_rel_table.dot(stable_pose.r[2,:])

        # make sure not in collision and above z
        if not in_collision and dist_to_table > IzzyState.DELTA_Z:
            grasp_set.append(grasp_candidate)
        i = i+1

    return grasp_set

# Grasp execution main function
def test_grasp_physical_success(graspable, grasp, stable_pose, dataset, config):
    debug = config['debug']
    load_object = config['load_object']
    alpha = config['alpha']
    center_scale = config['center_scale']
    tube_radius = config['tube_radius']
    table_extent = config['table_extent']
    lift_height = config['lift_height']

    # check collisions
    logging.info('Checking grasp collisions with table')
    grasp = grasp.grasp_aligned_with_stable_pose(stable_pose)
    debug_output = []
    does_collide = grasp.collides_with_stable_pose(stable_pose, debug_output)
    collision_box_vertices = np.array(debug_output[0]).T
    if does_collide:
        logging.error('Grasp is in collision')
        return

    # setup buffers
    exceptions = []
    actual_grasp_states = []
    target_grasp_states = []
    num_grasp_successes = 0
    num_grasp_lifts = 0
    num_grasp_trials = config['num_grasp_trials']
    grasp_dir = os.path.join(config['experiment_dir'], 'grasp_%d' %(grasp.grasp_id))
    if not os.path.exists(grasp_dir):
        os.mkdir(grasp_dir)

    # init hardware
    logging.info('Initializing hardware')
    camera = rs.RgbdSensor()
    ctrl = DexController()
    for i in range(num_grasp_trials):
        trial_start = time.time()

        logging.info('Grasp %d trial %d' %(grasp.grasp_id, i))
        logging_dir = os.path.join(grasp_dir, 'trial_%d' %(i))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)

        if True: #try:
            if not load_object:
                # move the arm out of the way
                logging.info('Moving arm out of the way')
                ctrl.reset_object()
                while not ctrl._izzy.is_action_complete():
                    time.sleep(0.01)

                # prompt for object placement
                yesno = raw_input('Place object. Hit [ENTER] when done')

                # retrieve object pose from camera
                logging.info('Registering object')
                depth_im = camera.get_depth_image()
                color_im = camera.get_color_image()
                registration_solver = tor.KnownObjectTabletopRegistrationSolver(logging_dir)
                reg_result = registration_solver.register(color_im, depth_im, graspable.key, dataset, config, debug=debug)
                T_camera_obj = reg_result.tf_camera_obj
                T_camera_obj.from_frame = 'obj'
                T_camera_obj.to_frame = 'camera'
                T_obj_camera = T_camera_obj.inverse()    
            else:
                # load the object pose from a file (for debugging only)
                T_camera_obj = stf.SimilarityTransform3D()
                T_camera_obj.load('data/calibration/spray_pose.stf')
                T_camera_obj.from_frame = 'obj'
                T_camera_obj.to_frame = 'camera'
                T_obj_camera = T_camera_obj.inverse()

            # transform the mesh to the stable pose to get a z offset from the table
            T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
            object_mesh = graspable.mesh
            object_mesh_tf = object_mesh.transform(T_obj_stp)
            #object_mesh_tf.visualize()
            #mv.axes()
            #mv.show()
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
            
            T_gripper_obj = grasp.gripper_transform(gripper='zeke')
            T_gripper_world = T_gripper_obj.dot(T_obj_world)

            # check gripper alignment (SPECIFIC TO THE SPRAY!)
            gripper_y_axis_world = T_gripper_world.rotation[:,1]
            obj_y_axis_world = T_obj_world.rotation[:,1]
            if gripper_y_axis_world.dot(obj_y_axis_world) > 0:
                R_gripper_p_gripper = np.array([[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]])
                T_gripper_p_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_gripper_p_gripper, np.zeros(3)), from_frame='gripper', to_frame='gripper')
                T_gripper_world = T_gripper_p_gripper.dot(T_gripper_world)

            # visualize the robot's understanding of the world
            logging.info('Displaying robot world state')
            mv.figure()
            mv_plot_table(T_table_world, d=table_extent)
            mv_plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mv_plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mv_plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mv_plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mv_plot_mesh(object_mesh, T_obj_world)
            mv_plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
            mv.show()

            # execute the grasp
            logging.info('Executing grasp')
            grasp_tf = T_gripper_world.inverse()
            ctrl.do_grasp(grasp_tf)
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            if debug:
                ctrl.plot_approach_angle()
    
            # record states
            current_state, _ = ctrl.getState()
            target_pose = DexRobotIzzy.IZZY_LOCAL_T * grasp_tf.pose
            target_pose.frame = DexConstants.IZZY_LOCAL_FRAME
            target_state = DexRobotIzzy.pose_to_state(target_pose, current_state)
            high_state = current_state.copy().set_arm_elev(lift_height)
            high_state.set_gripper_grip(IzzyState.MIN_STATE().gripper_grip)
            logging.info('Targeted state: %s' %(str(target_state)))
            logging.info('Reached state: %s' %(str(current_state)))

            target_state_dict = target_state.to_dict()
            target_state_dict['trial'] = i
            if i == 0:
                target_state_filename = os.path.join(grasp_dir, 'target_states.csv')
                target_f = open(target_state_filename, 'w')
                target_writer = csv.DictWriter(target_f, target_state_dict.keys())
                target_writer.writeheader()
            target_writer.writerow(target_state_dict)
            target_f.flush()

            current_state_dict = current_state.to_dict()
            current_state_dict['trial'] = i
            if i == 0:
                current_state_filename = os.path.join(grasp_dir, 'actual_states.csv')
                current_f = open(current_state_filename, 'w')
                current_writer = csv.DictWriter(current_f, current_state_dict.keys())
                current_writer.writeheader()
            current_writer.writerow(current_state_dict)
            current_f.flush()

            actual_grasp_states.append(current_state)
            target_grasp_states.append(target_state)

            # lift the object
            logging.info('Lifting object')
            ctrl._izzy.gotoState(high_state)
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            if debug:
                ctrl._izzy.plot()

            # get human input on grasp       
            human_input = raw_input('Did the grasp succeed? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded grasp success')
                num_grasp_successes = num_grasp_successes + 1
            else:
                logging.info('Recorded grasp failure')

            # get human input on grasp       
            human_input = raw_input('Did the grasp lift the object? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded lift success')
                num_grasp_lifts = num_grasp_lifts + 1
            else:
                logging.info('Recorded lift failure')

            # return to the reset state
            ctrl._izzy.gotoState(current_state)
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

            ctrl._izzy.unGrip()
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

            ctrl._izzy.reset()
            while not ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

        #except Exception as e:
        #    exceptions.append('Dataset: %s, Object: %s, Grasp: %d, Exception: %s' % (dataset.name, graspable.key, grasp.grasp_id, str(e)))

        trial_stop = time.time()
        logging.info('Trial %d took %f sec' %(i, trial_stop - trial_start))

    ctrl.stop()

    # compute the probability of success
    probability_success = float(num_grasp_successes) / float(num_grasp_trials)
    logging.info('Grasp %d probability of success: %f' %(grasp.grasp_id, probability_success))
    probability_lift = float(num_grasp_lifts) / float(num_grasp_trials)
    logging.info('Grasp %d probability of lift: %f' %(grasp.grasp_id, probability_lift))
    return probability_success, probability_lift

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    output_dir = sys.argv[2]

    # open config and read params
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    object_name = config['object_name']

    # open database and dataset
    database = db.Hdf5Database(database_filename, config)
    ds = database.dataset(dataset_name)

    # setup output directories and logging (TODO: make experiment wrapper class in future)
    experiment_id = 'single_grasp_experiment_%s' %(gen_experiment_id())
    experiment_dir = os.path.join(output_dir, experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    config['experiment_dir'] = experiment_dir
    experiment_log = os.path.join(experiment_dir, experiment_id +'.log')
    hdlr = logging.FileHandler(experiment_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr) 
    logging.info('RUNNING EXPERIMENT %s' %(experiment_id))

    # read the grasp metrics and features
    graspable = ds.graspable(object_name)
    grasps = ds.grasps(object_name)
    grasp_features = ds.grasp_features(object_name, grasps)
    grasp_metrics = ds.grasp_metrics(object_name, grasps)
    stable_poses = ds.stable_poses(object_name)
    stable_pose = stable_poses[config['stable_pose_index']]

    # compute the list of grasps to execute (TODO: update this section)
    grasps_to_execute = compute_grasp_set(ds, object_name, stable_pose, config['num_grasps_to_sample'], metric=config['grasp_metric'])

    # plot grasps
    for i, grasp in enumerate(grasps_to_execute):
        mv.clf()
        graspable.mesh.visualize(style='wireframe')
        T_gripper_obj = grasp.gripper_transform(gripper='zeke')
        mv_plot_pose(T_gripper_obj, alpha=config['alpha'], tube_radius=config['tube_radius'], center_scale=config['center_scale'])
        figname = 'grasp_%d.png' %(i)                
        mv.savefig(os.path.join(experiment_dir, figname))

    for i, grasp in enumerate(grasps_to_execute):
        logging.info('Evaluating grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps_to_execute)))

        # also compute deterministic metrics
        fc = quality.PointGraspMetrics3D.grasp_quality(grasp, graspable, method='force_closure')
        eps = quality.PointGraspMetrics3D.grasp_quality(grasp, graspable, method='ferrari_canny_L1')

        # output metrics in database
        grasp_q = grasp_metrics[grasp.grasp_id]
        grasp_q['grasp_id'] = grasp.grasp_id
        grasp_q['force_closure'] = fc
        grasp_q['ferrari_canny_l1'] = eps

        for metric in grasp_q.keys():
            logging.info('Quality according to %s: %f' %(metric, grasp_q[metric]))

        # get physical success ratio
        p_success, p_lift = test_grasp_physical_success(graspable, grasp, stable_pose, ds, config)
        grasp_q['p_grasp_success'] = p_success
        grasp_q['p_lift_success'] = p_lift

        # write output to CSV
        if i == 0:
            metric_filename = os.path.join(experiment_dir, 'grasp_metric_results.csv')
            output_f = open(metric_filename, 'w')
            writer = csv.DictWriter(output_f, grasp_q.keys())
            writer.writeheader()
        writer.writerow(grasp_q)
        output_f.flush()
