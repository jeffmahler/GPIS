# Import required Python code.
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


import IPython

NUM_COLLECT=20 # the number fo depth and pose csvs to collect
RGB_CAMERA='/camera/rgb'
DEPTH_CAMERA='/camera/depth_registered'
CAMERA_FRAME='/camera_rgb_optical_frame'
CB_FRAME='/chessboard_pose'

class GpisDataCollector(object):

    def __init__(self, rgb_camera, depth_camera, camera_frame, cb_frame):

        self.rgb_camera = rgb_camera
        self.depth_camera = depth_camera
        self.camera_frame = camera_frame
        self.cb_frame = cb_frame
        self.cb_topic = cb_frame

        # set up topic names
        rgb_image_topic = '%s/image_rect' %(self.rgb_camera)
        depth_image_topic = '%s/image' %(self.depth_camera)
        rgb_info_topic = '%s/camera_info' %(self.rgb_camera)  
        depth_info_topic = '%s/camera_info' %(self.depth_camera)

        # we will save the transform on each depth image
        self.listener = tf.TransformListener()
        self.poseSub = rospy.Subscriber(self.cb_topic, PoseStamped, self.pose_callback) 
        self.bridge = cv_bridge.CvBridge()

        # set up subscribers
        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.rgb_image_callback)
        self.depth_image_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_image_callback)
        self.rgb_info_sub = rospy.Subscriber(rgb_info_topic, CameraInfo, self.rgb_info_callback)
        self.depth_info_sub = rospy.Subscriber(depth_info_topic, CameraInfo, self.depth_info_callback)

        self.rgb_count = 0
        self.depth_count = 0
        self.pose_count = 0
        self.rgb_info_saved = False
        self.depth_info_saved = False

    def rgb_image_callback(self, msg):
        if self.rgb_count == 0:
            cv_rgb_image = self.bridge.imgmsg_to_cv(msg, "bgr8")

            # save grayscale version of image
            rgb_filename = 'rgb_%d.png' %(self.rgb_count)
            cv2.imwrite(rgb_filename, np.asarray(cv_rgb_image))

            self.rgb_count = self.rgb_count+1

    def depth_image_callback(self, msg):
        if self.depth_count < NUM_COLLECT:
            cv_depth_image = self.bridge.imgmsg_to_cv(msg, "32FC1")

            # save grayscale version of image
            depth_filename = 'depth_%d.csv' %(self.depth_count)
            depth_array = np.asarray(cv_depth_image)
            np.savetxt(depth_filename, depth_array, delimiter=',')

            self.depth_count = self.depth_count+1

    def pose_callback(self, msg):
        if self.pose_count < NUM_COLLECT:
            # save the transform from camera to chessboard
            t_filename = 'pr2_cb_transform_%d.csv' %(self.pose_count)
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            T = tfx.pose(position, quat)
            np.savetxt(t_filename, T.matrix, delimiter=',')

            self.pose_count = self.pose_count+1


    def rgb_info_callback(self, msg):
        if not self.rgb_info_saved:
            K_rgb_filename = 'K_rgb.csv'
            np.savetxt(K_rgb_filename, np.asarray(msg.K), delimiter=',')
            self.rgb_info_saved = True

    def depth_info_callback(self, msg):
        if not self.depth_info_saved:
            K_depth_filename = 'K_depth.csv'
            np.savetxt(K_depth_filename, np.asarray(msg.K), delimiter=',')
            self.depth_info_saved = True




if __name__ == '__main__':
    rospy.init_node('pr2_gpis_data_collection')

    gpd = GpisDataCollector(RGB_CAMERA, DEPTH_CAMERA, CAMERA_FRAME, CB_FRAME)

    rospy.spin()
