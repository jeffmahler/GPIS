"""
+X is front, +Y is left, +Z is up
"""
import copy
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np

import openravepy as rave

#from message_wrappers import GraspWrapper

import database as db
import grasp as g
import graspable_object
import obj_file
import sdf_file

import IPython

PR2_MODEL_FILE = 'data/models/pr2.robot.xml'

class OpenRaveGraspChecker(object):
    # global environment vars
    env_ = None
    robot_ = None

    def __init__(self, env = None, robot = None, view = True, win_height = 1200, win_width = 1200, cam_dist = 0.5):
        """ Defaults to using the PR2 """
        if env is None and (OpenRaveGraspChecker.env_ is None or OpenRaveGraspChecker.robot_ is None):
            OpenRaveGraspChecker._setup_rave_env()

        self.object_ = None
        self.view_ = view
        self._init_robot()        
        self._init_poses()
        if view:
            self._init_viewer(win_height, win_width, cam_dist)
        
    @property
    def env(self):
        if OpenRaveGraspChecker.env_ is None or OpenRaveGraspChecker.robot_ is None:
            OpenRaveGraspChecker._setup_rave_env()
        return OpenRaveGraspChecker.env_

    @property
    def robot(self):
        if OpenRaveGraspChecker.env_ is None or OpenRaveGraspChecker.robot_ is None:
            OpenRaveGraspChecker._setup_rave_env()
        return OpenRaveGraspChecker.robot_            

    @staticmethod
    def _setup_rave_env():
        """ OpenRave environment """
        OpenRaveGraspChecker.env_ = rave.Environment()
        OpenRaveGraspChecker.env_.Load(PR2_MODEL_FILE)
        OpenRaveGraspChecker.robot_ = OpenRaveGraspChecker.env_.GetRobots()[0]

    def _init_robot(self):
        """ Initialize the robot """
        # set initial pose
        self.robot.SetTransform(rave.matrixFromPose(np.array([1,0,0,0,0,0,0])))
        self.robot.SetDOFValues([0.54,-1.57, 1.57, 0.54],[22,27,15,34])
        
        # get robot manipulation tools
        self.manip_ = self.robot.SetActiveManipulator("leftarm_torso")
        self.maniprob_ = rave.interfaces.BaseManipulation(self.robot) # create the interface for task manipulation programs
        self.taskprob_ = rave.interfaces.TaskManipulation(self.robot) # create the interface for task manipulation programs
        self.finger_joint_ = self.robot.GetJoint('l_gripper_l_finger_joint')
        
    def _init_poses(self):
        """ Load the poses necessary for computing pregrasp poses """
        # pose from gripper to world frame
        self.T_gripper_world_ = self.robot.GetManipulator("leftarm_torso").GetTransform()

        # set transform between rviz and openrave
        R_fix = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
        T_fix = np.eye(4)
        T_fix[:3,:3] = R_fix
        T_fix[1,3] = 0
        T_fix[2,3] = -0.05
        self.T_rviz_or_ = T_fix

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

        # set only left robot gripper as visible
        for link in self.robot.GetLinks():
            link_name = link.GetName()
            if link_name[0] == u'l' and link_name.find('gripper') != -1:
                link.SetVisible(True)
            else:
                link.SetVisible(False)

    def _load_object(self, graspable_object):
        """ Load the object model into OpenRave """ 
        if graspable_object.model_name is None:
            raise ValueError('Graspable object model file name must be specified!')

        # load object model
        object_mesh_filename = graspable_object.model_name        
        if object_mesh_filename is None or object_mesh_filename == '':
            raise ValueError('Graspable must have a valid mesh filename to use OpenRave grasp checking')
        self.env.Load(object_mesh_filename)

        # intialize object, grasps
        obj = self.env.GetBodies()[1]
        return obj

    def get_link_mesh(self, link_name = 'l_gripper_palm_link'):
        """ Returns the vertices and triangles for the mesh of the given link """
        link = self.robot.GetLink(link_name)
        link_geoms = link.GetGeometries()
        link_tris = link_geoms[0].GetCollisionMesh()
        verts = link_tris.vertices
        inds = link_tris.indices
        return verts, inds

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
        self.robot.SetTransform(T_robot_world)

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

                # display finger closing if desired
                if close_fingers:
                    self.taskprob_.CloseFingers() # close fingers until collision
                    self.robot.WaitForController(0) # wait
                    time.sleep(1)
                    self.taskprob_.ReleaseFingers() # open fingets
                    self.robot.WaitForController(0) # wait
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
                    time.sleep(1)

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

def test_grasp_collisions():
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co.sdf'
#    sdf_3d_file_name = '/mnt/terastation/shape_data/MASTER_DB_v0/amazon_picking_challenge/dove_beauty_bar.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co.obj'
#    mesh_name = '/mnt/terastation/shape_data/MASTER_DB_v0/amazon_picking_challenge/dove_beauty_bar.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    grasp_checker = OpenRaveGraspChecker()

    center = np.array([0, 0, 0.02])
    axis = np.array([1, 0, 0]) 
    axis = axis / np.linalg.norm(axis)
    width = 0.1
    grasp = g.ParallelJawPtGrasp3D(center, axis, width)

    grasp.close_fingers(graspable, vis=True)
    grasp_checker.prune_grasps_in_collision(graspable, [grasp], auto_step=True, close_fingers=True, delay=30)

def test_grasp_poses():
    sdf_3d_file_name = '/mnt/terastation/shape_data/MASTER_DB_v0/amazon_picking_challenge/munchkin_white_hot_duck_bath_toy.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = '/mnt/terastation/shape_data/MASTER_DB_v0/amazon_picking_challenge/munchkin_white_hot_duck_bath_toy.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)
    center = np.array([0, 0, 0.02])
    axis = np.array([1, 0, 0])
    axis = axis / np.linalg.norm(axis)
    grasp = g.ParallelJawPtGrasp3D(center, axis, 0.1)
    rotated_grasps = grasp.transform(graspable.tf, 2 * np.pi / 10)

    grasp_checker = OpenRaveGraspChecker()
    rotated_grasps = grasp_checker.prune_grasps_in_collision(graspable, rotated_grasps, auto_step = True)

if __name__ == "__main__":
    test_grasp_collisions()
#    test_grasp_poses()
