import sys
sys.path.append("src/grasp_selection/control/DexControls")

import copy
import csv
import logging
import mayavi.mlab as mv
import IPython
from PIL import Image
import os
import numpy as np
import time
import tfx

from DexConstants import DexConstants
from DexController import DexController
from DexRobotIzzy import DexRobotIzzy
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from IzzyState import IzzyState

import camera_params as cp
import gripper as gr
import image_processing as ip
import mayavi_visualizer as mvis
import similarity_tf as stf
import tabletop_object_registration as tor

from objectives import Objective
from grasp import ParallelJawPtGrasp3D

class MABSingleObjectSoftHandObjective(Objective):

    def __init__(self, graspable, stable_pose, dataset, ctrl, camera, registration_solver, logging_dir, config):
        self.graspable = graspable
        self.stable_pose = stable_pose
        self.dataset = dataset
        self.ctrl = ctrl
        self.camera = camera
        self.registration_solver = registration_solver
        self.logging_dir = os.path.join(logging_dir, 'bandits')
        self.config = config

        self.num_evaluations = 0
        self.gripper = gr.RobotGripper.load(self.config['gripper'])

        if not os.path.exists(self.logging_dir):
            os.mkdir(self.logging_dir)

    # Grasp execution main function
    def evaluate(self, grasp):
        # read config
        debug = self.config['debug']
        load_object = self.config['load_object']
        alpha = self.config['alpha']
        center_scale = self.config['center_scale']
        tube_radius = self.config['tube_radius']
        table_extent = self.config['table_extent']
        lift_height = self.config['lift_height']
        num_grasp_views = self.config['num_grasp_views']
        cam_dist = self.config['cam_dist']
        num_avg_images = self.config['num_avg_images']
        output_dir = self.config['output_dir']
        threshold_cost = self.config['registration']['threshold_cost']
        hardware_reset = self.config['hardware_reset']
        table_low_thresh = self.config['table_low_thresh']
        table_high_thresh = self.config['table_high_thresh']
        calibration_dir = self.config['registration']['calibration_dir']
        focal_length = self.config['registration']['focal_length']
        depth_im_crop_dim = self.config['registration']['depth_im_crop_dim'] # the dimension of the cropped depth image

        graspable = self.graspable
        gripper = self.gripper
        camera = self.camera
        ctrl = self.ctrl
        registration_solver = self.registration_solver
        stable_pose = self.stable_pose

        # create logging dir
        logging_dir = os.path.join(self.logging_dir, 'trial_%d' %(self.num_evaluations))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)
        registration_solver.log_to(logging_dir)

        # run grasping trial
        trial_start = time.time()
        logging.info('Evaluating grasp %d' %(grasp.grasp_id))
        T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(self.stable_pose.r)) 

        table_rotation = 0
        registration_succeeded = False
        grasp_success = 0
        self.camera.reset()

        if True:#try:
            while not registration_succeeded:
                # move the arm out of the way
                logging.info('Moving arm out of the way')
                ctrl.reset_object()
        
                if hardware_reset == 1:
                    logging.info('Performing hardware reset')
                    ctrl._table.reset_fishing()
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
                camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], fx=focal_length)
                points_3d = camera_params.deproject(depth_im)
                subsample_inds = np.arange(points_3d.shape[1])[::10]
                points_3d = points_3d[:,subsample_inds]
                
                # register
                reg_result = registration_solver.register(copy.copy(color_im), copy.copy(depth_im), debug=debug)
                if reg_result.registration_results[reg_result.best_index].cost >= threshold_cost:
                    logging.info('Registration failed. Retrying...')
                    continue
                registration_succeeded = True
                
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
                        grasp_success = 1
                    else:
                        grasp_success = 0
                        logging.info('Recorded grasp failure')
                else:
                    # check object on table
                    depth_im_crop = ip.DepthImageProcessing.crop_center(depth_im, depth_im_crop_dim, depth_im_crop_dim)
                    camera_c_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], focal_length,
                                                      cx=depth_im_crop.shape[0]/2.0, cy=depth_im_crop.shape[1]/2.0)
                    points_3d = camera_c_params.deproject(depth_im_crop)

                    n = R_camera_table[:,2]
                    mean_point_plane = t_camera_table
                    x0_low = mean_point_plane + table_low_thresh * n.reshape(3,1)
                    x0_high = mean_point_plane + table_high_thresh * n.reshape(3,1)
                
                    points_3d_pruned, _ = ip.PointCloudProcessing.prune_points_above_plane(points_3d, n, x0_low)
                    points_3d_pruned, _ = ip.PointCloudProcessing.prune_points_above_plane(points_3d_pruned, -n, x0_high)

                    if points_3d_pruned.shape[1] == 0:
                        grasp_success = 1
                        logging.info('Grasp success')
                    else:
                        grasp_success = 0
                        logging.info('Grasp failure')
                
                # drop the object if picked up
                ctrl._robot.unGrip()
                reset_state = DexRobotZeke.RESET_STATES["OBJECT_RESET"]
                reset_state.set_arm_elev(lift_height)
                reset_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
                ctrl._robot.gotoState(reset_state)

                # log timings
                trial_stop = time.time()
                trial_duration = trial_stop - trial_start
                logging.info('Trial %d took %f sec' %(self.num_evaluations, trial_duration))

                # save results of trial
                grasp_trial_dict = {}
                grasp_trial_dict['object'] = graspable.key
                grasp_trial_dict['grasp_id'] = grasp.grasp_id
                grasp_trial_dict['trial'] = self.num_evaluations
                target_state_dict = target_state.to_dict()
                for key, val in target_state_dict.iteritems():
                    grasp_trial_dict['target_%s' %(key)] = val
                current_state_dict = current_state.to_dict()
                for key, val in current_state_dict.iteritems():
                    grasp_trial_dict['actual_%s' %(key)] = val
                misc_dict = {'success': grasp_success,
                             'force': sensors.gripper_force,
                             'grip_width': current_state.gripper_grip,
                             'table_rotation': table_rotation,
                             'duration': trial_duration,
                             'lifted_depth': lifted_depth_filename,
                             'lifted_color': lifted_color_filename,
                             'object_pose': obj_pose_filename,
                             'experiment_dir': self.config['experiment_dir']
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

        #except Exception as e:
        #    raise e

        self.num_evaluations += 1
        trial_stop = time.time()
        logging.info('Trial took %f sec' %(trial_stop - trial_start))

        # return successes and failures
        return grasp_success
    
    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, ParallelJawPtGrasp3D):
            raise ValueError("MAB Single Object objective can only be evaluated on ParallelJawPtGrasp3D objects")
