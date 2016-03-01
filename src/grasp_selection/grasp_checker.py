"""
+X is front, +Y is left, +Z is up
"""
import copy
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import openravepy as rave

#from message_wrappers import GraspWrapper

import database as db
import grasp as g
import graspable_object
import mesh_cleaner
import obj_file
import sdf_file
import similarity_tf as stf
import tfx

import IPython

PR2_MODEL_FILE = 'data/models/pr2.robot.xml'

class OpenRaveGraspChecker(object):
    # global environment vars
    env_ = None
    zeke_gripper_ = "data/robots/zeke_gripper/zeke_gripper.obj"
    zeke_gripper_t = "data/robots/zeke_gripper/T_grasp_to_gripper.stf"
    gripper_ = None
    t_grasp_to_gripper_ = None

    def __init__(self, env = None, gripper = None, t_grasp_to_gripper = None, view = True, win_height = 1200, win_width = 1200, cam_dist = 0.5):
        """ Defaults to using the Zeke gripper """
        if env is None and OpenRaveGraspChecker.env_ is None:
            OpenRaveGraspChecker._setup_rave_env()

        self.object_ = None
        self.view_ = view
        if view:
            self._init_viewer(win_height, win_width, cam_dist)
        self._init_gripper(gripper, t_grasp_to_gripper)

    @property
    def env(self):
        if OpenRaveGraspChecker.env_ is None:
            OpenRaveGraspChecker._setup_rave_env()
        return OpenRaveGraspChecker.env_

    @staticmethod
    def _setup_rave_env():
        """ OpenRave environment """
        OpenRaveGraspChecker.env_ = rave.Environment()

    def _init_gripper(self, gripper, t_grasp_to_gripper):
        if gripper is None or t_grasp_to_gripper is None:
            gripper = self._load_obj_from_file(OpenRaveGraspChecker.zeke_gripper_)
            t_grasp_to_gripper = stf.SimilarityTransform3D().load(OpenRaveGraspChecker.zeke_gripper_t)
        OpenRaveGraspChecker.gripper_ = gripper
        OpenRaveGraspChecker.t_grasp_to_gripper = t_grasp_to_gripper
     
    def _init_viewer(self, height, width, cam_dist):
        """ Initialize the OpenRave viewer """
        # set OR viewer
        OpenRaveGraspChecker.env_.SetViewer("qtcoin")
        viewer = self.env.GetViewer()
        viewer.SetSize(width, height)

        T_cam_obj = np.eye(4)
        R_cam_obj = np.array([[0,  0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
        T_cam_obj[:3,:3] = R_cam_obj
        T_cam_obj[0,3] = -cam_dist
        self.T_cam_obj_ = T_cam_obj

        # set view based on object
        self.T_obj_world_ = np.eye(4)
        self.T_cam_world_ = self.T_obj_world_.dot(self.T_cam_obj_)
        viewer.SetCamera(self.T_cam_world_, cam_dist)

    def _load_obj_from_file(self, filename):
        if filename is None or filename == '':
            raise ValueError('Object to load must have a valid mesh filename to use OpenRave grasp checking')
        self.env.Load(filename)

        obj = self.env.GetBodies()[-1]
        return obj    

    def _load_object(self, graspable_object):
        """ Load the object model into OpenRave """ 
        if graspable_object.model_name is None:
            raise ValueError('Graspable object model file name must be specified!')

        # load object model
        object_mesh_filename = graspable_object.model_name        
        return self._load_obj_from_file(object_mesh_filename)

    def move_to_pregrasp(self, grasp_pose, eps=1e-2):
        """ Move the robot to the pregrasp pose given by the grasp object """
        # get grasp pose
        gripper_position = grasp_pose.position
        gripper_orientation = grasp_pose.orientation
        gripper_pose = np.array([gripper_orientation.w, gripper_orientation.x, gripper_orientation.y, gripper_orientation.z, gripper_position.x, gripper_position.y, gripper_position.z])

        if abs(np.linalg.norm(np.array(gripper_orientation.list)) - 1.0) > eps:
            logging.warning('Illegal pose')
            return None, None
 
        # get grasp pose relative to object
        T_gripper_obj = rave.matrixFromPose(gripper_pose)
        T_obj_world = self.T_gripper_world_.dot(self.T_rviz_or_).dot(np.linalg.inv(T_gripper_obj))

        # set robot position as inverse of object (for viewing purposes)
        T_robot_world = np.linalg.inv(T_obj_world)
    
        return T_gripper_obj, T_robot_world
        
    def view_grasps(self, graspable, object_grasps, auto_step=False, close_fingers=False):
        """ Display all of the grasps """
        if self.env.GetViewer() is None and self.view_:
            self.env.SetViewer('qtcoin')
        ind = 0
        obj = self._load_object(graspable)

        for grasp in object_grasps:
            logging.info('Visualizing grasp %d' %(ind))
            self.move_to_pregrasp(grasp.gripper_pose())

            # only display grasps out of collision
            in_collision = self.env.CheckCollision(self.robot, obj)
            if not in_collision:
                if auto_step:
                    time.sleep(0.25)
                else:
                    user_input = 'x'
                    while user_input != '':
                        user_input = raw_input()

            ind = ind + 1

        self.env.Remove(obj)

    def prune_grasps_in_collision(self, graspable, object_grasps, closed_thresh = 0.05, auto_step = False, close_fingers = False, delay = 0.01):
        """ Remove all grasps from the object grasps list that are in collision with the given object """
        if self.env.GetViewer() is None and self.view_:
            self.env.SetViewer('qtcoin')
        ind = 0
        object_grasps_keep = []
        i = 0
        obj = self._load_object(graspable)

        # loop through grasps and check collisions for each
        for grasp in object_grasps:
            T_gripper_obj, T_robot_world = self.move_to_pregrasp(grasp.gripper_pose())
            # grasp.close_fingers(graspable, vis=True)
            # IPython.embed()

            if T_gripper_obj is None:
                continue

            # only display grasps out of collision
            in_collision = self.env.CheckCollision(self.robot, obj)
            time.sleep(1)
            if not in_collision:
                if auto_step:
                    time.sleep(delay)
                else:
                    user_input = 'x'
                    while user_input != '':
                        user_input = raw_input()

                # display finger closing if desired
                if close_fingers:
                    # close fingers until collision
                    self.taskprob_.CloseFingers()
                    self.robot.WaitForController(0) # wait

                    user_input = 'x'
                    while user_input != '':
                        user_input = raw_input()
                    #time.sleep(1)

                    # check that the gripper contacted something
                    closed_amount = self.finger_joint_.GetValues()[0]
                    if closed_amount > closed_thresh:
                        object_grasps_keep.append(grasp)

                    # open fingers
                    self.taskprob_.ReleaseFingers()
                    self.robot.WaitForController(0) # wait
                else:
                    object_grasps_keep.append(grasp)

            i = i+1

        self.env.Remove(obj)
        return object_grasps_keep

def test_grasp_collision():
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = '/mnt/terastation/shape_data/MASTER_DB_v2/Cat50_ModelDatabase/5c7bf45b0f847489181be2d6e974dccd.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = '/mnt/terastation/shape_data/MASTER_DB_v2/Cat50_ModelDatabase/5c7bf45b0f847489181be2d6e974dccd.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    clean_mesh_name = 'data/test/meshes/flashlight.obj'
    mc = mesh_cleaner.MeshCleaner(m)
    mc.rescale_vertices(0.07)
    oof = obj_file.ObjFile(clean_mesh_name)
    oof.write(mc.mesh())

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m, model_name=clean_mesh_name,
                                                   tf=stf.SimilarityTransform3D(tfx.identity_tf(), 0.75))

    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    grasp_checker = OpenRaveGraspChecker()

    center = np.array([0, 0.01, 0])
    axis = np.array([1, 0, 0]) 
    axis = axis / np.linalg.norm(axis)
    width = 0.1
    grasp = g.ParallelJawPtGrasp3D(g.ParallelJawPtGrasp3D.configuration_from_params(center, axis, width))
    
    rotated_grasps = grasp.transform(graspable.tf, 2.0 * np.pi / 20.0)
    
    IPython.embed()

    '''
    print len(rotated_grasps)
    grasp_checker.prune_grasps_in_collision(graspable, rotated_grasps, auto_step=False, close_fingers=True, delay=1)
    '''

if __name__ == "__main__":
    test_grasp_collision()
