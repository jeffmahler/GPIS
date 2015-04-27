"""
+X is front, +Y is left, +Z is up
"""
import os
import sys
import time

import numpy as np
import openravepy as rave
import IK as ik
from pymongo import MongoClient

from message_wrappers import GraspWrapper

import IPython

APC_DIRECTORY = os.path.abspath(os.path.join(__file__, "../.."))
DATA_DIRECTORY = os.path.join(APC_DIRECTORY, "data")

PR2_MODEL_FILE = "data/models/pr2.robot.xml"
OBJECT_MESH_DIR = "data/meshes/objects"

class PR2GraspChecker(object):
    def __init__(self, env, robot, object_name, win_height = 1200, win_width = 1200, cam_dist = 0.5):
        self.env = env
        self.robot = robot

        self.loadObject(object_name)
        self.initRobot()        
        self.initPoses()
        self.initViewer(win_height, win_width, cam_dist)

    def loadObject(self, object_name):
        # construct object filename
        object_filename = os.path.join(OBJECT_MESH_DIR, object_name + '.stl')
        object_grasps_filename = os.path.join(DATA_DIRECTORY, 'grasps',
                                              "{}.json".format(object_name))

        # load object model
        e.Load(object_filename)

        # intialize object, grasps
        self.object = e.GetBodies()[1]
        self.object_grasps = GraspWrapper.grasps_from_file(object_grasps_filename)
        
    def initRobot(self):
        # set initial pose
        self.robot.SetTransform(rave.matrixFromPose(np.array([1,0,0,0,0,0,0])))
        self.robot.SetDOFValues([0.54,-1.57, 1.57, 0.54],[22,27,15,34])
        
        # get robot manipulation tools
        self.manip = self.robot.SetActiveManipulator("leftarm_torso")
        self.maniprob = rave.interfaces.BaseManipulation(self.robot) # create the interface for task manipulation programs
        self.taskprob = rave.interfaces.TaskManipulation(self.robot) # create the interface for task manipulation programs
        self.finger_joint = self.robot.GetJoint('l_gripper_l_finger_joint')
        
    def initPoses(self):
        # pose from gripper to world frame
        self.T_gripper_world = self.robot.GetManipulator("leftarm_torso").GetTransform()

        # set transform between rviz and openrave
        R_fix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        T_fix = np.eye(4)
        T_fix[:3,:3] = R_fix
        T_fix[1,3] = 0.008
        T_fix[2,3] = -0.0375
        self.T_fix = T_fix

    def initViewer(self, height, width, cam_dist):
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

    def moveToPregrasp(self, grasp):
        # get grasp pose
        gripper_position = grasp.gripper_pose.position
        gripper_orientation = grasp.gripper_pose.orientation
        gripper_pose = np.array([gripper_orientation.w, gripper_orientation.x, gripper_orientation.y, gripper_orientation.z, gripper_position.x, gripper_position.y, gripper_position.z])

        # get grasp pose relative to object
        T_gripper_obj = rave.matrixFromPose(gripper_pose)
        T_obj_world = self.T_gripper_world.dot(self.T_fix).dot(np.linalg.inv(T_gripper_obj))

        # set robot position as inverse of object (for viewing purposes)
        T_robot_world = np.linalg.inv(T_obj_world)
        self.robot.SetTransform(T_robot_world)

        return T_gripper_obj, T_robot_world
        
    def viewGrasps(self, auto_step, close_fingers):
        '''
        Only display all of the grasps
        '''
        if self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        ind = 0
        for grasp in self.object_grasps:
            self.moveToPregrasp(grasp)

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

    def pruneBadGrasps(self, vis = True, closed_thresh = 0.05, auto_step = True, close_fingers = True):
        '''
        Only display all of the grasps
        '''
        if vis and self.env.GetViewer() is None:
            self.env.SetViewer('qtcoin')
        ind = 0
        object_grasps_keep = []
        i = 0

        for grasp in self.object_grasps:
            if vis:
                T_gripper_obj, T_robot_world = self.moveToPregrasp(grasp)

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
                    # close fingers until collision
                    self.taskprob.CloseFingers()
                    self.robot.WaitForController(0) # wait
                    time.sleep(1)

                    # check that the gripper contacted something
                    closed_amount = self.finger_joint.GetValues()[0]
                    if closed_amount > closed_thresh:
                        object_grasps_keep.append(grasp)
                    else:
                        print 'BAD GRASP'
                        #IPython.embed()

                    # open fingers
                    self.taskprob.ReleaseFingers()
                    self.robot.WaitForController(0) # wait
                else:
                    object_grasps_keep.append(grasp)

            i = i+1

        return object_grasps_keep
    
if __name__ == "__main__":
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

