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
        if not in_collision and dist_to_table > ZekeState.DELTA_Z:
            grasp_set.append(grasp_candidate)
        i = i+1
    
    return grasp_set

# Grasp execution main function
def test_grasp_physical_success(graspable, grasp, stable_pose, dataset, registration_solver, ctrl, camera, config):
    debug = config['debug']
    load_object = config['load_object']
    alpha = config['alpha']
    center_scale = config['center_scale']
    tube_radius = config['tube_radius']
    table_extent = config['table_extent']
    lift_height = config['lift_height']
    num_grasp_views = config['num_grasp_views']
    cam_dist = config['cam_dist']

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
    num_grasp_trials = config['num_grasp_trials']
    grasp_successes = np.zeros(num_grasp_trials)
    grasp_lifts = np.zeros(num_grasp_trials)
    grasp_dir = os.path.join(config['experiment_dir'], 'grasp_%d' %(grasp.grasp_id))
    if not os.path.exists(grasp_dir):
        os.mkdir(grasp_dir)

    # run grasping trials
    for i in range(num_grasp_trials):
        trial_start = time.time()

        logging.info('Grasp %d trial %d' %(grasp.grasp_id, i))
        logging_dir = os.path.join(grasp_dir, 'trial_%d' %(i))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)
        registration_solver.log_to(logging_dir)

        if True:#try:
            if not load_object:
                # move the arm out of the way
                logging.info('Moving arm out of the way')
                ctrl.reset_object()

                # prompt for object placement
                yesno = raw_input('Place object. Hit [ENTER] when done')

                # retrieve object pose from camera
                logging.info('Registering object')
                depth_im = camera.get_depth_image()
                color_im = camera.get_color_image()
                
                # get point cloud (for debugging only)
                camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], fx=config['registration']['focal_length'])
                points_3d = camera_params.deproject(depth_im)
                subsample_inds = np.arange(points_3d.shape[1])[::20]
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
            
            T_gripper_obj = grasp.gripper_transform(gripper='zeke')
            T_gripper_world = T_gripper_obj.dot(T_obj_world)

            # check gripper alignment (SPECIFIC TO THE SPRAY!)
            gripper_y_axis_world = T_gripper_world.rotation[1,:]
            obj_y_axis_world = T_obj_world.rotation[1,:]
            if gripper_y_axis_world.dot(obj_y_axis_world) > 0:
                logging.info('Flipping grasp')
                R_gripper_p_gripper = np.array([[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]])
                T_gripper_p_gripper = stf.SimilarityTransform3D(pose=tfx.pose(R_gripper_p_gripper, np.zeros(3)), from_frame='gripper', to_frame='gripper')
                T_gripper_world = T_gripper_p_gripper.dot(T_gripper_world)

            # visualize the robot's understanding of the world
            logging.info('Displaying robot world state')
            mv.clf()
            mvis.MayaviVisualizer.plot_table(T_table_world, d=table_extent)
            mvis.MayaviVisualizer.plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_mesh(object_mesh, T_obj_world)
            mvis.MayaviVisualizer.plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
            #mvis.MayaviVisualizer.plot_point_cloud(points_3d, T_world_camera, color=(0,1,1))
            mv.show()
            
            delta_view = 360.0 / num_grasp_views
            mv.view(distance=cam_dist)
            for j in range(num_grasp_views):
                az = j * delta_view
                mv.view(azimuth=az)
                figname = 'estimated_scene_view_%d.png' %(j)                
                mv.savefig(os.path.join(logging_dir, figname))

            # execute the grasp
            logging.info('Executing grasp')
            grasp_tf = T_gripper_world.inverse()
            target_state = ctrl.do_grasp(grasp_tf)
            if debug:
                ctrl.plot_approach_angle()
    
            # record states
            current_state, _ = ctrl.getState()
            high_state = current_state.copy().set_arm_elev(lift_height)
            high_state.set_gripper_grip(ZekeState.MIN_STATE().gripper_grip)
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
            ctrl._robot.gotoState(high_state)
            if debug:
                ctrl._robot.plot()

            # get human input on grasp success
            human_input = raw_input('Did the grasp succeed? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded grasp success')
                grasp_successes[i] = 1
            else:
                logging.info('Recorded grasp failure')

            # get human input on grasp lift
            human_input = raw_input('Did the grasp lift the object? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded lift success')
                grasp_lifts[i] = 1
            else:
                logging.info('Recorded lift failure')

            success_dict = {'trial': i, 'success': grasp_successes[i], 'lift': grasp_lifts[i]}
            if i == 0:
                success_filename = os.path.join(grasp_dir, 'grasp_successes.csv')
                success_f = open(success_filename, 'w')
                success_writer = csv.DictWriter(success_f, success_dict.keys())
                success_writer.writeheader()
            success_writer.writerow(success_dict)
            success_f.flush()

            # return to the reset state
            ctrl._robot.gotoState(current_state)

            ctrl._robot.unGrip()

            reset_state = DexRobotZeke.RESET_STATES["GRIPPER_SAFE_RESET"]
            reset_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
            ctrl._robot.gotoState(reset_state)

        #except Exception as e:
        #    logging.error(str(e))
        #    exceptions.append('Dataset: %s, Object: %s, Grasp: %d, Exception: %s' % (dataset.name, graspable.key, grasp.grasp_id, str(e)))

        trial_stop = time.time()
        logging.info('Trial %d took %f sec' %(i, trial_stop - trial_start))

    # log all exceptions
    exceptions_filename = os.path.join(grasp_dir, 'exceptions.txt')
    out_exceptions = open(exceptions_filename, 'w')
    for exception in exceptions:
        out_exceptions.write('%s\n' %(exception))
    out_exceptions.close()

    # return successes and failures
    return grasp_successes, grasp_lifts

if __name__ == '__main__':
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

    # copy over config file
    config_path, config_fileroot = os.path.split(config_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_fileroot))
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
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
    object_mesh = graspable.mesh
    object_mesh_tf = object_mesh.transform(T_obj_stp)
    delta_view = 360.0 / num_grasp_views

    for i, grasp in enumerate(grasps_to_execute):
        logging.info('Grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps_to_execute)))
        mv.clf()
        object_mesh_tf.visualize(style='wireframe')
        mvis.MayaviVisualizer.plot_grasp(grasp, T_obj_stp, alpha=1.5*config['alpha'], tube_radius=config['tube_radius'])

        for j in range(num_grasp_views):
            az = j * delta_view
            mv.view(az)
            figname = 'grasp_%d_view_%d.png' %(i, j)                
            mv.savefig(os.path.join(experiment_dir, figname))

    # preload registration solver
    registration_solver = tor.KnownObjectTabletopRegistrationSolver(object_name, ds, config)

    # init hardware
    logging.info('Initializing camera')
    camera = rs.RgbdSensor()
    logging.info('Initializing robot')
    ctrl = DexController()

    """
    current_state, _ = ctrl.getState()
    logging.info('Before: %s' %(str(current_state)))

    reset_state = DexRobotZeke.RESET_STATES['GRIPPER_SAFE_RESET']
    reset_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
    reset_state.set_arm_ext(None)
    reset_state.set_arm_elev(None)
    ctrl._robot.gotoState(reset_state)
    time.sleep(6)

    current_state, _ = ctrl.getState()
    logging.info('After: %s' %(str(current_state)))

    #ctrl._robot.grip()
    ctrl.stop()
    #IPython.embed()
    exit(0)
    """

    # run bandits to debug
    """
    objective = msoo.MABSingleObjectObjective(graspable, stable_pose, ds, ctrl, camera, config)
    ts = das.ThompsonSampling(objective, grasps_to_execute)
    logging.info('Running Thompson sampling.')

    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]
    ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)
    ctrl.stop()
    """

    # execute each grasp
    for i, grasp in enumerate(grasps_to_execute):
        if True:#try:
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

            # reset camera
            camera.reset()

            # get physical success ratio
            grasp_successes, grasp_lifts = test_grasp_physical_success(graspable, grasp, stable_pose, ds, registration_solver,
                                                                       ctrl, camera, config)

            # compute the probability of success
            p_success = np.mean(grasp_successes)
            p_lift = np.mean(grasp_lifts)
            grasp_q['p_grasp_success'] = p_success
            grasp_q['p_lift_success'] = p_lift
            logging.info('Grasp %d probability of success: %f' %(grasp.grasp_id, p_success))
            logging.info('Grasp %d probability of lift: %f' %(grasp.grasp_id, p_lift))

            # write output to CSV
            if i == 0:
                metric_filename = os.path.join(experiment_dir, 'grasp_metric_results.csv')
                output_f = open(metric_filename, 'w')
                writer = csv.DictWriter(output_f, grasp_q.keys())
                writer.writeheader()
            writer.writerow(grasp_q)
            output_f.flush()
        #except Exception as e:
            # TODO: handle more gracefully
        #    ctrl.stop()
        #    raise e

    ctrl.stop()
