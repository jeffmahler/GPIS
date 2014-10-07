import roslib
import argparse
roslib.load_manifest('raven_2_vision')
import rospy

import tf
import tfx
import cv
import cv2
import cv_bridge
import numpy as np
from scipy import linalg
import math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo

from pr2 import arm
from pr2_sim import simulator
from utils import utils

import IPython

GRASP_DIR='/home/peter/jeff_working/raven_2/raven_2_vision/grasps/pfc'
GRASP_NAME='pc_tape'
CENTER_NAME='center'

PR2_BASE_FRAME='base_link'
R_ARM_FRAME='r_gripper_tool_frame'
L_ARM_FRAME='l_gripper_tool_frame'

DIRECTION_NAME='dir'
ENDPOINTS_NAME='endpoints'
CB_TOPIC='/chessboard_pose'

class Pr2Grasper(object):

    def __init__(self, cb_topic, grasp_dir, grasp_name):
        
        self.cb_topic = cb_topic
        self.grasp_dir = grasp_dir
        self.grasp_name = grasp_name

        self.grasp_initial_z_offset = 0.15 # position to initialize grasp
        self.grasp_final_z_offset = 0.075 # move grasp up a tad
        self.grasp_pickup_z_offset = 0.3 # final position

        self.pr2_base_frame=PR2_BASE_FRAME
        self.r_arm_frame = R_ARM_FRAME
        self.l_arm_frame = L_ARM_FRAME
        self.T_cb_cam = None
        self.grasp_msg = None

        self.pose_sub = rospy.Subscriber(self.cb_topic, PoseStamped, self.pose_callback) 
        self.gripper_pub = rospy.Publisher('/gpis_grasp_pose', PoseStamped)  # for debugging

        # load in the grasp attributes and convert to end effector pose
        self.load_grasp()
        self.compute_grasp_pose()

        # open up a PR2 interface
        sim = None
        self.r_arm = arm.Arm('right', sim=sim, default_speed=0.05)
        self.l_arm = arm.Arm('left', sim=sim, default_speed=0.05)
    
    def load_grasp(self):
        # load in the center, direction, endpoints csv files
        center_filename = '%s/%s_%s.csv' %(self.grasp_dir, self.grasp_name, CENTER_NAME)
        direction_filename = '%s/%s_%s.csv' %(self.grasp_dir, self.grasp_name, DIRECTION_NAME)
        endpoints_filename = '%s/%s_%s.csv' %(self.grasp_dir, self.grasp_name, ENDPOINTS_NAME)
        
        # read the NP arrays
        self.grasp_center = np.loadtxt(open(center_filename, 'r'), delimiter=',')
        self.grasp_direction = np.loadtxt(open(direction_filename, 'r'), delimiter=',')
        self.grasp_endpoints = np.loadtxt(open(endpoints_filename, 'r'), delimiter=',')

        self.grasp_setup_center = np.copy(self.grasp_center)
        self.grasp_pickup_center = np.copy(self.grasp_center)

        # offset the grasp center
        self.grasp_center[2] = self.grasp_center[2] + self.grasp_final_z_offset
        self.grasp_setup_center[2] = self.grasp_setup_center[2] + self.grasp_initial_z_offset
        self.grasp_pickup_center[2] = self.grasp_pickup_center[2] + self.grasp_pickup_z_offset

    def compute_grasp_pose(self):
        # wait for the chessboard pose
        timeout = utils.Timeout(10)
        timeout.start()
        while self.T_cb_cam is None and not timeout.has_timed_out():
            rospy.sleep(0.5)

        # convert grasp center and direction to the pr2 frame
        grasp_dir_cam = self.T_cb_cam[:3,:3].dot(self.grasp_direction)
        grasp_center_cam = self.T_cb_cam.matrix.dot(self.grasp_center)
        grasp_setup_center_cam = self.T_cb_cam.matrix.dot(self.grasp_setup_center)
        grasp_pickup_center_cam = self.T_cb_cam.matrix.dot(self.grasp_pickup_center)

        # convert center, dir to a pose
        grasp_x_axis = np.array(-1 * self.T_cb_cam[:3,2]) # into the chessboard
        grasp_y_axis = np.array(grasp_dir_cam.T)
        grasp_x_axis = grasp_x_axis[:,0]
        grasp_y_axis = grasp_y_axis[:,0]        

        grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)

        R_grasp_cam = np.c_[grasp_x_axis, grasp_y_axis]
        R_grasp_cam = np.c_[R_grasp_cam, grasp_z_axis]

        t_grasp_cam = grasp_center_cam
        t_grasp_cam = np.array(t_grasp_cam[0,:3])

        T_grasp_cam = np.c_[R_grasp_cam, t_grasp_cam.T]
        T_grasp_cam = np.r_[T_grasp_cam, np.array([[0, 0, 0, 1]])]

        self.pose_grasp_cam = tfx.pose(T_grasp_cam, frame=self.T_cb_cam.frame)
        quat_grasp_cam = self.pose_grasp_cam.orientation

        self.grasp_msg = PoseStamped()
        self.grasp_msg.header.stamp = rospy.Time.now()
        self.grasp_msg.header.frame_id = self.T_cb_cam.frame

        self.grasp_msg.pose.position.x = T_grasp_cam[0,3]
        self.grasp_msg.pose.position.y = T_grasp_cam[1,3]
        self.grasp_msg.pose.position.z = T_grasp_cam[2,3]

        self.grasp_msg.pose.orientation.x = quat_grasp_cam[0]
        self.grasp_msg.pose.orientation.y = quat_grasp_cam[1]
        self.grasp_msg.pose.orientation.z = quat_grasp_cam[2]
        self.grasp_msg.pose.orientation.w = quat_grasp_cam[3]

        # get the pose in the gripper frame
        self.pose_grasp_base_link = tfx.convertToFrame(self.pose_grasp_cam, self.pr2_base_frame, self.pose_grasp_cam.frame, wait=10)

        # form grasp setup pose
        t_grasp_setup_cam = grasp_setup_center_cam
        t_grasp_setup_cam = np.array(t_grasp_setup_cam[0,:3])

        T_grasp_setup_cam = np.c_[R_grasp_cam, t_grasp_setup_cam.T]
        T_grasp_setup_cam = np.r_[T_grasp_setup_cam, np.array([[0, 0, 0, 1]])]
        self.pose_grasp_setup_cam = tfx.pose(T_grasp_setup_cam, frame=self.T_cb_cam.frame)
        self.pose_grasp_setup_base_link = tfx.convertToFrame(self.pose_grasp_setup_cam, self.pr2_base_frame, self.pose_grasp_setup_cam.frame, wait=10)

        # form pickup pose
        t_grasp_pickup_cam = grasp_pickup_center_cam
        t_grasp_pickup_cam = np.array(t_grasp_pickup_cam[0,:3])

        T_grasp_pickup_cam = np.c_[R_grasp_cam, t_grasp_pickup_cam.T]
        T_grasp_pickup_cam = np.r_[T_grasp_pickup_cam, np.array([[0, 0, 0, 1]])]
        self.pose_grasp_pickup_cam = tfx.pose(T_grasp_pickup_cam, frame=self.T_cb_cam.frame)
        self.pose_grasp_pickup_base_link = tfx.convertToFrame(self.pose_grasp_pickup_cam, self.pr2_base_frame, self.pose_grasp_pickup_cam.frame, wait=10)


    def pose_callback(self, msg):
        if self.T_cb_cam is None:
            # convert the pose msg to a tfx pose
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

            self.T_cb_cam = tfx.pose(position, quat, frame=msg.header.frame_id)
   
        if self.grasp_msg is not None:
            self.gripper_pub.publish(self.grasp_msg)

    def attempt_grasp(self):
        self.r_arm.open_gripper(block=True)

        # go slightly above the object
        self.r_arm.go_to_pose(self.pose_grasp_setup_base_link, block=True)

        # move to grasp position and close
        self.r_arm.go_to_pose(self.pose_grasp_base_link, block=True)
        self.r_arm.close_gripper(block=True)

        # lift up object
        self.r_arm.go_to_pose(self.pose_grasp_pickup_base_link, block=True)

       
if __name__ == '__main__':
    rospy.init_node('pr2_gpis_grasping')

    pgr = Pr2Grasper(CB_TOPIC, GRASP_DIR, GRASP_NAME)
    pgr.attempt_grasp()

    rospy.spin()
