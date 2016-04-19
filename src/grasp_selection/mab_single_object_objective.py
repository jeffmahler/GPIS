import sys
sys.path.append("src/grasp_selection/control/DexControls")

import copy
import csv
import logging
import mayavi.mlab as mv
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
import mayavi_visualizer as mvis
import similarity_tf as stf
import tabletop_object_registration as tor

from objectives import Objective
from grasp import ParallelJawPtGrasp3D

class MABSingleObjectObjective(Objective):

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

        # create logging dir
        logging_dir = os.path.join(self.logging_dir, 'trial_%d' %(self.num_evaluations))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)

        # run grasping trial
        trial_start = time.time()
        logging.info('Evaluating grasp %d' %(grasp.grasp_id))
        T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(self.stable_pose.r)) 

        grasp_success = 0
        self.camera.reset()

        if True: #try:
            # move arm out of the way
            logging.info('Moving arm out of the way')
            self.registration_solver.log_to(self.logging_dir)
            self.ctrl.reset_object()

            # prompt for object placement
            yesno = raw_input('Place object. Hit [ENTER] when done')

            # retrieve object pose from self.camera
            logging.info('Registering object')
            depth_im = self.camera.get_depth_image()
            color_im = self.camera.get_color_image()
                
            # get point cloud (for debugging only)
            camera_params = cp.CameraParams(depth_im.shape[0], depth_im.shape[1], fx=self.config['registration']['focal_length'])
            points_3d = camera_params.deproject(depth_im)
            subsample_inds = np.arange(points_3d.shape[1])[::20]
            points_3d = points_3d[:,subsample_inds]

            # register
            reg_result = self.registration_solver.register(copy.copy(color_im), copy.copy(depth_im), debug=debug)
            T_camera_obj = reg_result.tf_camera_obj
            T_camera_obj.from_frame = 'obj'
            T_camera_obj.to_frame = 'self.camera'
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
            T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(self.stable_pose.r)) 
            object_mesh = self.graspable.mesh
            object_mesh_tf = object_mesh.transform(T_obj_stp)
            mn, mx = object_mesh_tf.bounding_box()
            z = mn[2]
    
            # define poses of self.camera, table, object, tec
            T_world = stf.SimilarityTransform3D(from_frame='world', to_frame='world')
            R_table_world = np.eye(3)
            T_table_world = stf.SimilarityTransform3D(pose=tfx.pose(R_table_world, np.zeros(3)), from_frame='world', to_frame='table')
            
            R_camera_table = np.load('data/calibration/rotation_camera_cb.npy')
            t_camera_table = np.load('data/calibration/translation_camera_cb.npy')
            cb_points_camera = np.load('data/calibration/corners_cb.npy')
            T_camera_table = stf.SimilarityTransform3D(tfx.pose(R_camera_table, t_camera_table), from_frame='table', to_frame='self.camera')
            T_camera_world = T_camera_table.dot(T_table_world)
            T_world_camera = T_camera_world.inverse()
    
            R_stp_obj = self.stable_pose.r
            T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')
            
            t_stp_table = np.array([0, 0, z])
            T_stp_table = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), t_stp_table), from_frame='table', to_frame='stp')
    
            T_obj_world = T_obj_camera.dot(T_camera_world)
            
            T_gripper_obj = grasp.gripper_transform(gripper=self.gripper)
            T_gripper_world = T_gripper_obj.dot(T_obj_world)

            # visualize the robot's understanding of the world
            logging.info('Displaying robot world state')
            mv.clf()
            mvis.MayaviVisualizer.plot_table(T_table_world, d=table_extent)
            mvis.MayaviVisualizer.plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            mvis.MayaviVisualizer.plot_mesh(object_mesh, T_obj_world, color=(1,0,0))
            mvis.MayaviVisualizer.plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))
            mvis.MayaviVisualizer.plot_gripper(grasp, T_obj_world, self.gripper)

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
            target_state = self.ctrl.do_grasp(grasp_tf)
            if debug:
                self.ctrl.plot_approach_angle()
                plt.show()

            # record states
            current_state, _ = self.ctrl.getState()
            high_state = current_state.copy().set_arm_elev(lift_height)
            high_state.set_gripper_grip(ZekeState.MIN_STATE().gripper_grip)
            logging.info('Targeted state: %s' %(str(target_state)))
            logging.info('Reached state: %s' %(str(current_state)))

            target_state_dict = target_state.to_dict()
            target_state_dict['grasp'] = grasp.grasp_id
            target_state_filename = os.path.join(logging_dir, 'target_states.csv')
            target_f = open(target_state_filename, 'w')
            target_writer = csv.DictWriter(target_f, target_state_dict.keys())
            target_writer.writeheader()
            target_writer.writerow(target_state_dict)
            target_f.flush()

            current_state_dict = current_state.to_dict()
            current_state_dict['grasp'] = grasp.grasp_id
            current_state_filename = os.path.join(logging_dir, 'actual_states.csv')
            current_f = open(current_state_filename, 'w')
            current_writer = csv.DictWriter(current_f, current_state_dict.keys())
            current_writer.writeheader()
            current_writer.writerow(current_state_dict)
            current_f.flush()

            # lift the object
            logging.info('Lifting object')
            self.ctrl._robot.gotoState(high_state)
            sensors = self.ctrl._robot.getSensors()
            if debug:
                self.ctrl._robot.plot()

            # save image of lifted object for future use
            depth_im = self.camera.get_depth_image()
            min_d = np.min(depth_im)
            max_d = np.max(depth_im)
            depth_im2 = 255.0 * (depth_im - min_d) / (max_d - min_d)
            depth_im2 = Image.fromarray(depth_im2.astype(np.uint8))
            filename = 'lifted_depth.png'
            depth_im2.save(os.path.join(logging_dir, filename))

            # get human input on grasp success
            human_input = raw_input('Did the grasp succeed? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded grasp success')
                grasp_success = 1
            else:
                logging.info('Recorded grasp failure')

            # drop the object
            self.ctrl._robot.gotoState(current_state)
            self.ctrl._robot.unGrip()

            # go to reset
            reset_state = DexRobotZeke.RESET_STATES["GRIPPER_SAFE_RESET"]
            reset_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
            self.ctrl._robot.gotoState(reset_state)

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
