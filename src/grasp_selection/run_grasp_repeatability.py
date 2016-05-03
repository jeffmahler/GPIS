"""
Script to evaluate the probability of success for a few grasps on Zeke, logging the target states and the predicted quality in simulation
Authors: Jeff Mahler and Jacky Liang
"""
import copy
import csv
import logging
import IPython
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mayavi.mlab as mv
import numpy as np
import pickle as pkl
import openravepy as rave
import random
from random import choice
import os
import shutil
import sys
sys.path.append("src/grasp_selection/control/DexControls")
import time

from DexAngles import DexAngles
from DexConstants import DexConstants
from DexController import DexController
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from TurntableState import TurntableState

import camera_params as cp
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import grasp_collision_checker as gcc
import gripper as gr
import mab_single_object_objective as msoo
import mayavi_visualizer as mvis
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
def test_grasp_physical_success(graspable, grasp, gripper, stable_pose, dataset, registration_solver, ctrl, camera, config):
    debug = config['debug']
    load_object = config['load_object']
    alpha = config['alpha']
    center_scale = config['center_scale']
    tube_radius = config['tube_radius']
    table_extent = config['table_extent']
    lift_height = config['lift_height']
    num_grasp_views = config['num_grasp_views']
    cam_dist = config['cam_dist']
    num_avg_images = config['num_avg_images']
    output_dir = config['output_dir']
    threshold_cost = config['registration']['threshold_cost']
    hardware_reset = config['hardware_reset']
    table_rotation = 0

    # check collisions
    logging.info('Checking grasp collisions with table')
    grasp = grasp.grasp_aligned_with_stable_pose(stable_pose)

    # setup buffers
    exceptions = []
    actual_grasp_states = []
    target_grasp_states = []
    num_grasp_trials = config['num_grasp_trials']
    grasp_successes = np.zeros(num_grasp_trials)
    grasp_lifts = np.zeros(num_grasp_trials)
    grasp_dir = os.path.join(config['experiment_dir'], 'grasp_%d' %(grasp.grasp_id))
    if not os.path.exists(grasp_dir):
        os.mkdir(grasp_dir)

    # run grasping trials
    i = 0
    while i < num_grasp_trials:
        trial_start = time.time()

        # setup logging
        logging.info('Grasp %d trial %d' %(grasp.grasp_id, i))
        logging_dir = os.path.join(grasp_dir, 'trial_%d' %(i))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)
        registration_solver.log_to(logging_dir)

        # reset camera
        camera.reset()

        try:
            if not load_object:
                # move the arm out of the way
                logging.info('Moving arm out of the way')
                ctrl.reset_object()
        
                if hardware_reset == 1:
                    ctrl._table.reset_fishing()
                    time.sleep(1.0) # to prevent blurring
                else:
                    # prompt for object placement
                    yesno = raw_input('Place object. Hit [ENTER] when done')

                    # rotate table a random amount
                    table_state = ctrl._table.getState()
                    table_rotation = 2 * np.pi * np.random.rand() - np.pi
                    table_state.set_table_rot(table_rotation + table_state.table_rot)
                    logging.info('Rotating table by %f' %(table_rotation))
                    ctrl._table.gotoState(table_state)

                # retrieve object pose from camera
                logging.info('Registering object')
                depth_im = np.zeros([camera.height, camera.width])
                counts = np.zeros([camera.height, camera.width])
                for k in range(num_avg_images):
                    new_depth_im = camera.get_depth_image()
                    color_im = camera.get_color_image()

                    depth_im = depth_im + new_depth_im
                    counts = counts + np.array(new_depth_im > 0.0)
                depth_im[depth_im > 0] = depth_im[depth_im > 0] / counts[depth_im > 0]
                
                # get point cloud (for debugging only)
                camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], fx=config['registration']['focal_length'])
                points_3d = camera_params.deproject(depth_im)
                subsample_inds = np.arange(points_3d.shape[1])[::10]
                points_3d = points_3d[:,subsample_inds]

                # register
                reg_result = registration_solver.register(copy.copy(color_im), copy.copy(depth_im), debug=debug)
                if reg_result.registration_results[reg_result.best_index].cost >= threshold_cost:
                    logging.info('Registration failed. Retrying...')
                    continue
                
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
            
            T_gripper_obj = grasp.gripper_transform(gripper=gripper)
            T_gripper_world = T_gripper_obj.dot(T_obj_world)

            obj_pose_filename = os.path.join(logging_dir, 'T_obj_world.stf')
            T_obj_world.save(obj_pose_filename)

            # visualize the robot's understanding of the world
            logging.info('Displaying robot world state')
            mv.clf()
            mvis.MayaviVisualizer.plot_table(T_table_world, d=table_extent)
            mvis.MayaviVisualizer.plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_mesh(object_mesh, T_obj_world, color=(1,0,0))
            mvis.MayaviVisualizer.plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
            mvis.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper)
            #mvis.MayaviVisualizer.plot_point_cloud(points_3d, T_world_camera, color=(0,1,0), scale=0.0025)
            #mv.view(focalpoint=(0,0,0))
            #mv.show()

            delta_view = 360.0 / num_grasp_views
            for j in range(num_grasp_views):
                az = j * delta_view
                mv.view(azimuth=az, focalpoint=(0,0,0), distance=cam_dist)
                figname = 'estimated_scene_view_%d.png' %(j)                
                mv.savefig(os.path.join(logging_dir, figname))

            # execute the grasp
            logging.info('Executing grasp')
            grasp_tf = T_gripper_world.inverse()
            target_state = ctrl.do_grasp(grasp_tf)
            if debug:
                ctrl.plot_approach_angle()
                plt.show()

            # record states
            current_state, _ = ctrl.getState()
            high_state = current_state.copy().set_arm_elev(lift_height)
            high_state.set_gripper_grip(ZekeState.MIN_STATE().gripper_grip)
            logging.info('Targeted state: %s' %(str(target_state)))
            logging.info('Reached state: %s' %(str(current_state)))

            actual_grasp_states.append(current_state)
            target_grasp_states.append(target_state)

            # lift the object
            logging.info('Lifting object')
            ctrl._robot.gotoState(high_state)
            sensors = ctrl._robot.getSensors()
            if debug:
                ctrl._robot.plot()

            # save image of lifted object for future use
            color_im = camera.get_color_image()
            lifted_color_filename = os.path.join(logging_dir, 'lifted_color.png')
            color_im = Image.fromarray(color_im.astype(np.uint8))
            color_im.save(lifted_color_filename)

            depth_im = camera.get_depth_image()
            min_d = np.min(depth_im)
            max_d = np.max(depth_im)
            depth_im2 = 255.0 * (depth_im - min_d) / (max_d - min_d)
            depth_im2 = Image.fromarray(depth_im2.astype(np.uint8))
            lifted_depth_filename = os.path.join(logging_dir, 'lifted_depth.png')
            depth_im2.save(lifted_depth_filename)

            # get human input on grasp success
            if hardware_reset == 0:
                human_input = raw_input('Did the grasp succeed? [y/n] ')
                while human_input.lower() != 'n' and human_input.lower() != 'y':
                    logging.info('Did not understand input. Please answer \'y\' or \'n\'')
                    human_input = raw_input('Did the grasp succeed? [y/n] ')

                if human_input.lower() == 'y':
                    logging.info('Recorded grasp success')
                    grasp_successes[i] = 1
                else:
                    logging.info('Recorded grasp failure')

                # get human input on grasp lift
                human_input = raw_input('Did the grasp lift the object? [y/n] ')
                while human_input.lower() != 'n' and human_input.lower() != 'y':
                    logging.info('Did not understand input. Please answer \'y\' or \'n\'')
                    human_input = raw_input('Did the grasp lift the object? [y/n] ')

                if human_input.lower() == 'y':
                    logging.info('Recorded lift success')
                    grasp_lifts[i] = 1
                else:
                    logging.info('Recorded lift failure')

            # drop the object if picked up
            ctrl._robot.unGrip()
            reset_state = DexRobotZeke.RESET_STATES["OBJECT_RESET"]
            reset_state.set_arm_elev(lift_height)
            reset_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
            ctrl._robot.gotoState(reset_state)

            # log timings
            trial_stop = time.time()
            trial_duration = trial_stop - trial_start
            logging.info('Trial %d took %f sec' %(i, trial_duration))

            # save results of trial
            grasp_trial_dict = {}
            grasp_trial_dict['trial'] = i
            target_state_dict = target_state.to_dict()
            for key, val in target_state_dict.iteritems():
                grasp_trial_dict['target_%s' %(key)] = val
            current_state_dict = current_state.to_dict()
            for key, val in current_state_dict.iteritems():
                grasp_trial_dict['actual_%s' %(key)] = val
            misc_dict = {'success': grasp_successes[i],
                         'lift': grasp_lifts[i], 
                         'force': sensors.gripper_force,
                         'grip_width': current_state.gripper_grip,
                         'table_rotation': table_rotation,
                         'duration': trial_duration,
                         'lifted_depth': lifted_depth_filename,
                         'lifted_color': lifted_color_filename,
                         'object_pose': obj_pose_filename,
                         'experiment_dir': config['experiment_dir']
                         }
            grasp_trial_dict.update(misc_dict)

            grasp_trial_filename = os.path.join(output_dir, 'grasp_trial_results.csv')
            if os.path.exists(grasp_trial_filename):
                output_f = open(grasp_trial_filename, 'a')
                writer = csv.DictWriter(output_f, grasp_trial_dict.keys())
            else:
                output_f = open(grasp_trial_filename, 'w')
                writer = csv.DictWriter(output_f, grasp_trial_dict.keys())
                writer.writeheader()
            writer.writerow(grasp_trial_dict)
            output_f.flush()
            
        except Exception as e:
            logging.error(str(e))
            exceptions.append('Dataset: %s, Object: %s, Grasp: %d, Exception: %s' % (dataset.name, graspable.key, grasp.grasp_id, str(e)))
        
        # check for skip
        if hardware_reset == 0:
            human_input = raw_input('Continue with grasp?')
            if human_input.lower() == 'no':
                logging.info('Skipping grasp')
                break

        i += 1

    # log all exceptions
    exceptions_filename = os.path.join(grasp_dir, 'exceptions.txt')
    out_exceptions = open(exceptions_filename, 'w')
    for exception in exceptions:
        out_exceptions.write('%s\n' %(exception))
    out_exceptions.close()

    # return successes and failures
    return grasp_successes, grasp_lifts

if __name__ == '__main__':
    test = False
    if test:
        np.random.seed(100)
        random.seed(100)

    # read params
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

    # init viewer
    mv.figure(size=(1000, 1000), bgcolor=(0.8, 0.8, 0.8))

    # open database and dataset
    database = db.Hdf5Database(database_filename, config)
    ds = database.dataset(dataset_name)
    gripper = gr.RobotGripper.load(config['gripper'])

    # setup output directories and logging (TODO: make experiment wrapper class in future)
    experiment_id = 'single_grasp_experiment_%s' %(gen_experiment_id())
    experiment_dir = os.path.join(output_dir, experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    config['output_dir'] = output_dir
    config['experiment_dir'] = experiment_dir
    experiment_log = os.path.join(experiment_dir, experiment_id +'.log')
    hdlr = logging.FileHandler(experiment_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr) 

    # add to master csv
    experiment_id_dict = {'experiment_id': experiment_id, 'use': 0}
    filename = os.path.join(output_dir, 'experiment_directory.csv')
    if os.path.exists(filename):
        f = open(filename, 'a')
        csv_writer = csv.DictWriter(f, experiment_id_dict.keys())
    else:
        f = open(filename, 'w')
        csv_writer = csv.DictWriter(f, experiment_id_dict.keys())
        csv_writer.writeheader()
    csv_writer.writerow(experiment_id_dict)
    f.flush()

    # copy over config file
    dexconstants_filename = 'src/grasp_selection/control/DexControls/DexConstants.py'
    zekestate_filename = 'src/grasp_selection/control/DexControls/ZekeState.py'
    turntablestate_filename = 'src/grasp_selection/control/DexControls/TurntableState.py'
    config_path, config_fileroot = os.path.split(config_filename)
    dexconstants_path, dexconstants_fileroot = os.path.split(dexconstants_filename)
    zekestate_path, zekestate_fileroot = os.path.split(zekestate_filename)
    turntablestate_path, turntablestate_fileroot = os.path.split(turntablestate_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_fileroot))
    shutil.copyfile(dexconstants_filename, os.path.join(experiment_dir, dexconstants_fileroot))
    shutil.copyfile(zekestate_filename, os.path.join(experiment_dir, zekestate_fileroot))
    shutil.copyfile(turntablestate_filename, os.path.join(experiment_dir, turntablestate_fileroot))
    logging.info('RUNNING EXPERIMENT %s' %(experiment_id))

    # read the grasp metrics and features
    graspable = ds.graspable(object_name)
    grasps = ds.grasps(object_name, gripper=config['gripper'])
    grasp_metrics = ds.grasp_metrics(object_name, grasps, gripper=config['gripper'])
    stable_poses = ds.stable_poses(object_name)
    stable_pose = stable_poses[config['stable_pose_index']]
    
    # HACK to fix a y-axis orientation bug in the stable pose code
    if np.abs(np.linalg.det(stable_pose.r) + 1) < 0.01:
        stable_pose.r[1,:] = -stable_pose.r[1,:]

    # load the list of grasps to execute
    candidate_grasp_dir = os.path.join(config['grasp_candidate_dir'], ds.name)
    grasp_id_filename = os.path.join(candidate_grasp_dir,
                                     '%s_%s_%s_grasp_ids.npy' %(object_name, stable_pose.id, config['gripper']))
    grasp_ids = np.load(grasp_id_filename)
    grasps_to_execute = []
    for grasp in grasps:
        if grasp.grasp_id in grasp_ids and grasp.grasp_id >= config['start_grasp_id']:
            grasps_to_execute.append(grasp.grasp_aligned_with_stable_pose(stable_pose))
            break

    # plot grasps
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
    T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')
    object_mesh = graspable.mesh
    object_mesh_tf = object_mesh.transform(T_obj_stp)
    delta_view = 360.0 / num_grasp_views

    for i, grasp in enumerate(grasps_to_execute):
        logging.info('Grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps_to_execute)))
        mv.clf()
        T_obj_world = mvis.MayaviVisualizer.plot_stable_pose(graspable.mesh, stable_pose, T_table_world, d=0.1,
                                                             style='surface', color=(1,0,0))
        mvis.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper=gripper)

        for j in range(num_grasp_views):
            az = j * delta_view
            mv.view(az)
            figname = 'grasp_%d_view_%d.png' %(grasp.grasp_id, j)                
            mv.savefig(os.path.join(experiment_dir, figname))

    # preload registration solver
    registration_solver = tor.KnownObjectStablePoseTabletopRegistrationSolver(object_name, stable_pose.id, ds, config)

    # init hardware
    logging.info('Initializing Camera')
    camera = rs.RgbdSensor()
    ctrl = DexController()

    # execute each grasp
    for i, grasp in enumerate(grasps_to_execute):
        try:
            logging.info('Evaluating grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps_to_execute)))

            # output metrics in database
            grasp_q = grasp_metrics[grasp.grasp_id]
            grasp_q['grasp_id'] = grasp.grasp_id

            for metric in grasp_q.keys():
                logging.info('Quality according to %s: %f' %(metric, grasp_q[metric]))

            # reset camera
            camera.reset()

            # get physical success ratio
            grasp_successes, grasp_lifts = test_grasp_physical_success(graspable, grasp, gripper,
                                                                       stable_pose, ds,
                                                                       registration_solver,
                                                                       ctrl, camera, config)

            # compute the probability of success
            p_success = np.mean(grasp_successes)
            p_lift = np.mean(grasp_lifts)
            grasp_q['p_grasp_success'] = p_success
            grasp_q['p_lift_success'] = p_lift
            logging.info('Grasp %d probability of success: %f' %(grasp.grasp_id, p_success))
            logging.info('Grasp %d probability of lift: %f' %(grasp.grasp_id, p_lift))

            # write output to CSV
            metric_filename = os.path.join(output_dir, 'grasp_metric_results.csv')
            if os.path.exists(metric_filename):
                output_f = open(metric_filename, 'a')
                writer = csv.DictWriter(output_f, grasp_q.keys())
            else:
                output_f = open(metric_filename, 'w')
                writer = csv.DictWriter(output_f, grasp_q.keys())
                writer.writeheader()
            writer.writerow(grasp_q)
            output_f.flush()
        except Exception as e:
            # TODO: handle more gracefully
            ctrl.stop()
            raise e

    ctrl.stop()

    # run bandits to debug
    """
    objective = msoo.MABSingleObjectObjective(graspable, stable_pose, ds, ctrl, camera,
                                              registration_solver, experiment_dir, config)
    ts = das.ThompsonSampling(objective, grasps_to_execute)
    logging.info('Running Thompson sampling.')

    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]
    ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)
    ctrl.stop()

    # save bandit results
    f = open(os.path.join(experiment_dir, 'results.pkl'), 'w')
    pkl.dump(ts_result, f)
    exit(0)
    """
