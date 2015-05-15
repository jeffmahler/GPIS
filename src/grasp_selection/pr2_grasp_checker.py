"""
+X is front, +Y is left, +Z is up
"""
import copy
import os
import sys
import time

import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np
import openravepy as rave
import IK as ik
from pymongo import MongoClient

from message_wrappers import GraspWrapper

import grasp as g
import graspable_object
import obj_file
import sdf_file

import IPython

PR2_MODEL_FILE = 'data/models/pr2.robot.xml'

class PR2GraspChecker(object):
    def __init__(self, env, robot, object_mesh_filename, win_height = 1200, win_width = 1200, cam_dist = 0.5):
        self.env = env
        self.robot = robot

        self._load_object(object_mesh_filename)
        self._init_robot()        
        self._init_poses()
        self._init_viewer(win_height, win_width, cam_dist)

    def _load_object(self, object_mesh_filename):
        """ Load the object model into OpenRave """ 
        # load object model
        self.env.Load(object_mesh_filename)

        # intialize object, grasps
        self.object = self.env.GetBodies()[1]
        
    def _init_robot(self):
        """ Initialize the robot """
        # set initial pose
        self.robot.SetTransform(rave.matrixFromPose(np.array([1,0,0,0,0,0,0])))
        self.robot.SetDOFValues([0.54,-1.57, 1.57, 0.54],[22,27,15,34])
        
        # get robot manipulation tools
        self.manip = self.robot.SetActiveManipulator("leftarm_torso")
        self.maniprob = rave.interfaces.BaseManipulation(self.robot) # create the interface for task manipulation programs
        self.taskprob = rave.interfaces.TaskManipulation(self.robot) # create the interface for task manipulation programs
        self.finger_joint = self.robot.GetJoint('l_gripper_l_finger_joint')
        
    def _init_poses(self):
        """ Load the poses necessary for computing pregrasp poses """
        # pose from gripper to world frame
        self.T_gripper_world = self.robot.GetManipulator("leftarm_torso").GetTransform()

        # set transform between rviz and openrave
        R_fix = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
        T_fix = np.eye(4)
        T_fix[:3,:3] = R_fix
        T_fix[1,3] = 0
        T_fix[2,3] = -0.0375
        self.T_rviz_or = T_fix

    def _init_viewer(self, height, width, cam_dist):
        """ Initialize the OpenRave viewer """
        # set OR viewer
        viewer = self.env.GetViewer()
        viewer.SetSize(width, height)

        T_cam_obj = np.eye(4)
        R_cam_obj = np.array([[0,  0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
        T_cam_obj[:3,:3] = R_cam_obj
        T_cam_obj[0,3] = -cam_dist
        self.T_cam_obj = T_cam_obj

        # set view based on object
        self.T_obj_world = np.eye(4)
        self.T_cam_world = self.T_obj_world.dot(self.T_cam_obj)
        viewer.SetCamera(self.T_cam_world, cam_dist)

        # set only left robot gripper as visible
        for link in self.robot.GetLinks():
            link_name = link.GetName()
            if link_name[0] == u'l' and link_name.find('gripper') != -1:
                link.SetVisible(True)
            else:
                link.SetVisible(False)
            #link.SetVisible(True)

    def move_to_pregrasp(self, grasp_pose):
        """ Move the robot to the pregrasp pose given by the grasp object """
        # get grasp pose
        gripper_position = grasp_pose.position
        gripper_orientation = grasp_pose.orientation
        gripper_pose = np.array([gripper_orientation.w, gripper_orientation.x, gripper_orientation.y, gripper_orientation.z, gripper_position.x, gripper_position.y, gripper_position.z])

        # get grasp pose relative to object
        T_gripper_obj = rave.matrixFromPose(gripper_pose)
        T_obj_world = self.T_gripper_world.dot(self.T_rviz_or).dot(np.linalg.inv(T_gripper_obj))

        # set robot position as inverse of object (for viewing purposes)
        T_robot_world = np.linalg.inv(T_obj_world)
        self.robot.SetTransform(T_robot_world)

        return T_gripper_obj, T_robot_world
        
    def view_grasps(self, auto_step, close_fingers):
        '''
        Only display all of the grasps
        '''
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        ind = 0
        for grasp in self.object_grasps:
            self.move_to_pregrasp(grasp)

            # only display grasps out of collision
            in_collision = self.env.CheckCollision(self.robot, self.object)
            if not in_collision:
                if auto_step:
                    time.sleep(0.2)
                else:
                    user_input = 'x'
                    while user_input != '':
                        user_input = raw_input()

                # display finger closing if desired
                if close_fingers:
                    self.taskprob.CloseFingers() # close fingers until collision
                    self.robot.WaitForController(0) # wait
                    time.sleep(1)
                    self.taskprob.ReleaseFingers() # open fingets
                    self.robot.WaitForController(0) # wait

    def prune_grasps_in_collision(self, object_grasps, vis = True, closed_thresh = 0.05, auto_step = False, close_fingers = False):
        """ Remove all grasps from the object grasps list that are in collision with the given object """
        if vis and self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        ind = 0
        object_grasps_keep = []
        i = 0

        for grasp in object_grasps:
            pose = grasp.gripper_pose()
            if vis:
                T_gripper_obj, T_robot_world = self.move_to_pregrasp(pose)
            """
            link = self.robot.GetLink('l_gripper_palm_link')
            link_geoms = link.GetGeometries()
            link_tris = link_geoms[0].GetCollisionMesh()
            verts = link_tris.vertices
            inds = link_tris.indices
            IPython.embed()
            """
            # only display grasps out of collision
            in_collision = self.env.CheckCollision(self.robot, self.object)
            if not in_collision:
                if auto_step:
                    time.sleep(0.1)
                else:
                    user_input = 'x'
                    while user_input != '':
                        user_input = raw_input()

                # display finger closing if desired
                if close_fingers:
                    # close fingers until collision
                    self.taskprob.CloseFingers()
                    self.robot.WaitForController(0) # wait
                    time.sleep(1)

                    # check that the gripper contacted something
                    closed_amount = self.finger_joint.GetValues()[0]
                    if closed_amount > closed_thresh:
                        object_grasps_keep.append(grasp)

                    # open fingers
                    self.taskprob.ReleaseFingers()
                    self.robot.WaitForController(0) # wait
                else:
                    object_grasps_keep.append(grasp)

            i = i+1

        return object_grasps_keep

def test_grasp_collisions():
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m)

    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    e = rave.Environment()
    e.Load(PR2_MODEL_FILE)
    e.SetViewer("qtcoin")
    r = e.GetRobots()[0]
    grasp_checker = PR2GraspChecker(e, r, mesh_name)

    """
    z = np.linspace(-0.1, 0.1, 50)
    T = grasp_checker.object.GetTransform()
    for i in range(z.shape[0]):
        print z
        T_tf = copy.copy(T)
        T_tf[2,3] = z[i]
        grasp_checker.object.SetTransform(T_tf)
        time.sleep(0.1)

    for i in range(z.shape[0]):
        print z
        T_tf = copy.copy(T)
        T_tf[1,3] = z[i]
        grasp_checker.object.SetTransform(T_tf)
        time.sleep(0.1)

    for i in range(z.shape[0]):
        print z
        T_tf = copy.copy(T)
        T_tf[0,3] = z[i]
        grasp_checker.object.SetTransform(T_tf)
        time.sleep(0.1)
    """

    center = np.array([-0.01837182,  0.01218656,  0.03844461])
    # np.array([-0.00297651,  0.00677959,  0.03165122])
    axis = np.array([-0.94212793,  0.31667146, -0.11006428]) 
    #np.array([1, 1, 0])
    axis = axis / np.linalg.norm(axis)
    grasp = g.ParallelJawPtGrasp3D(center, axis, 0.1)

    grasp.close_fingers(graspable, vis = True)
    grasp_checker.prune_grasps_in_collision([grasp], vis = True)

    IPython.embed()

if __name__ == "__main__":
    test_grasp_collisions()
    """
    # get sys input
    argc = len(sys.argv)
    if argc < 2:
        print 'Need to supply object name'
        exit(0)

    object_name = sys.argv[1]

    # load openrave environment
    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    e = rave.Environment()
    e.Load(PR2_MODEL_FILE)
    e.SetViewer("qtcoin")

    # set robot in front of the shelf
    r = e.GetRobots()[0]

    # prune grasps
    grasp_checker = PR2GraspChecker(e, r, object_name)
    object_grasps_keep = grasp_checker.pruneBadGrasps()

    # resave the json file
    object_grasps_out_filename = os.path.join(DATA_DIRECTORY, 'grasps',
                                              "{}.json".format(object_name + '_coll_free'))
    GraspWrapper.grasps_to_file(object_grasps_keep, object_grasps_out_filename)

    """
