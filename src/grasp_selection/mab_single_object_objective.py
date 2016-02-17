import sys
sys.path.append("src/grasp_selection/control/DexControls")

import os
import numpy as np
import time

import similarity_tf as stf
import tabletop_object_registration as tor
import mayavi.mlab as mv
import tfx
import logging
import copy
from DexConstants import DexConstants

from objectives import Objective
from grasp import ParallelJawPtGrasp3D
from MayaviVisualizer import MayaviVisualizer

class MABSingleObjectObjective(Objective):

    def __init__(self, graspable, stable_pose, dataset, ctrl, camera, config):
        self.graspable = graspable
        self.stable_pose = stable_pose
        self.dataset = dataset
        self.ctrl = ctrl
        self.camera = camera
        self.config = config

    # Grasp execution main function
    def evaluate(self, grasp):
        grasp_success = 0
    
        debug = self.config['debug']
        load_object = self.config['load_object']
        alpha = self.config['alpha']
        center_scale = self.config['center_scale']
        tube_radius = self.config['tube_radius']
        table_extent = self.config['table_extent']
        lift_height = self.config['lift_height']
        num_grasp_views = self.config['num_grasp_views']
        cam_dist = self.config['cam_dist']

        # check collisions
        logging.info('Checking grasp collisions with table')
        grasp = grasp.grasp_aligned_with_stable_pose(self.stable_pose)
        debug_output = []
        does_collide = grasp.collides_with_stable_pose(self.stable_pose, debug_output)
        collision_box_vertices = np.array(debug_output[0]).T
        if does_collide:
            logging.error('Grasp is in collision')
            return

        # setup buffers
        exceptions = []
        grasp_dir = os.path.join(self.config['experiment_dir'], 'grasp_%d' %(grasp.grasp_id))
        if not os.path.exists(grasp_dir):
            os.mkdir(grasp_dir)

        # run grasping trial
        trial_start = time.time()

        logging.info('Grasp %d' %(grasp.grasp_id))
        logging_dir = os.path.join(grasp_dir, 'trial_%d' %(0))
        if not os.path.exists(logging_dir):
            os.mkdir(logging_dir)

        try:
            if not load_object:
                # move the arm out of the way
                logging.info('Moving arm out of the way')
                self.ctrl.reset_object()
                while not self.ctrl._izzy.is_action_complete():
                    time.sleep(0.01)

                # prompt for object placement
                yesno = raw_input('Place object. Hit [ENTER] when done')

                # retrieve object pose from camera
                logging.info('Registering object')
                depth_im = self.camera.get_depth_image()
                color_im = self.camera.get_color_image()
                registration_solver = tor.KnownObjectTabletopRegistrationSolver(logging_dir)
                reg_result = registration_solver.register(copy.copy(color_im), copy.copy(depth_im), self.graspable.key, self.dataset, self.config, debug=debug)
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
            T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(self.stable_pose.r)) 
            object_mesh = self.graspable.mesh
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
    
            R_stp_obj = self.stable_pose.r
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
            MayaviVisualizer.mv_plot_table(T_table_world, d=table_extent)
            MayaviVisualizer.mv_plot_pose(T_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            MayaviVisualizer.mv_plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            MayaviVisualizer.mv_plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            MayaviVisualizer.mv_plot_pose(T_camera_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
            MayaviVisualizer.mv_plot_mesh(object_mesh, T_obj_world)
            MayaviVisualizer.mv_plot_point_cloud(cb_points_camera, T_world_camera, color=(1,1,0))

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
            self.ctrl.do_grasp(grasp_tf)
            while not self.ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            if debug:
                self.ctrl.plot_approach_angle()
    
            # record states
            current_state, _ = self.ctrl.getState()
            target_pose = DexRobotIzzy.IZZY_LOCAL_T * grasp_tf.pose
            target_pose.frame = DexConstants.IZZY_LOCAL_FRAME
            target_state = DexRobotIzzy.pose_to_state(target_pose, current_state)
            high_state = current_state.copy().set_arm_elev(lift_height)
            high_state.set_gripper_grip(IzzyState.MIN_STATE().gripper_grip)
            logging.info('Targeted state: %s' %(str(target_state)))
            logging.info('Reached state: %s' %(str(current_state)))

            # lift the object
            logging.info('Lifting object')
            self.ctrl._izzy.gotoState(high_state)
            while not self.ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            if debug:
                self.ctrl._izzy.plot()

            # get human input on grasp success
            human_input = raw_input('Did the grasp succeed? [y/n] ')
            if human_input.lower() != 'n':
                logging.info('Recorded grasp success')
                grasp_success = 1
            else:
                logging.info('Recorded grasp failure')

            # return to the reset state
            self.ctrl._izzy.gotoState(current_state)
            while not self.ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

            self.ctrl._izzy.unGrip()
            while not self.ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

            reset_state = DexRobotIzzy.RESET_STATES["GRIPPER_SAFE_RESET"]
            reset_state.set_gripper_grip(IzzyState.MAX_STATE().gripper_grip)
            self.ctrl._izzy.gotoState(reset_state)
            while not self.ctrl._izzy.is_action_complete():
                time.sleep(0.01)
            time.sleep(2)

        except Exception as e:
            logging.error(str(e))
            exceptions.append('Dataset: %s, Object: %s, Grasp: %d, Exception: %s' % (self.dataset.name, self.graspable.key, grasp.grasp_id, str(e)))

        trial_stop = time.time()
        logging.info('Trial took %f sec' %(trial_stop - trial_start))

        # log all exceptions
        exceptions_filename = os.path.join(grasp_dir, 'exceptions.txt')
        out_exceptions = open(exceptions_filename, 'w')
        for exception in exceptions:
            out_exceptions.write('%s\n' %(exception))
        out_exceptions.close()

        # return successes and failures
        return grasp_success
    
    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, ParallelJawPtGrasp3D):
            raise ValueError("MAB Single Object objective can only be evaluated on ParallelJawPtGrasp3D objects")