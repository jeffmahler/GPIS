'''
Encapsulates camera parameters for projecting / deprojecitng points
Author: Jeff Mahler
'''
import IPython
import numpy as np
import os
from PIL import Image, ImageDraw
import sklearn.decomposition
import sys
import tfx

class CameraParams:
        '''
        Encapsulates camera parameters and the operations we want to do with them
        '''
        def __init__(self, height, width, fx, fy=None, cx=None, cy=None, pose=tfx.identity_tf()):
                '''
                Init camera parameters
                
                Params:
                   height: (int or float) height of image
                   width: (int of flaot) width of image
                   fx: (float) x focal length of camera in pixels
                   fy: (float) y focal length of camera in pixels
                   cx: (float) optical center of camera in pixels along x axis
                   cy: (float) optical center of camera in pixels along y axis
                '''
                self.height_ = height
                self.width_ = width
                self.fx_ = fx
                self.pose_ = pose

                # set focal, camera center automatically if under specified
                if fy is None:
                        self.fy_ = fx
                else:
                        self.fy_= fy
                if cx is None:
                        self.cx_ = float(width) / 2
                else:
                        self.cx_ = cx
                if cy is None:
                        self.cy_ = float(height) / 2
                else:
                        self.cy_ = cy
                # set camera projection matrix
                self.K_ = np.array([[self.fx_,        0, self.cx_],
                                    [       0, self.fy_, self.cy_],
                                    [       0,        0,        1]])

        def height(self):
                return self.height_

        def width(self):
                return self.width_

        def proj_matrix(self):
                return self.K_
        
        def pose(self):
                return self.pose_

        def project(self, points):
                '''
                Projects a set of points into the camera given by these parameters
                
                Params:
                   points: (3xN numpy array of floats) 3D points to project
                Returns:
                   2xN numpy float array of 2D image coordinates
                   1xN binary numpy array indicating whether or not point projected outside of image
                '''
                # check valid data
                if points.shape[0] != 3:
                        raise Exception('Incorrect data dimension. CameraParams project must be supplied a 3xN numpy float array.')

                points_proj = self.K_.dot(points)
                point_depths = np.tile(points_proj[2,:], [3, 1])
                points_proj = np.divide(points_proj, point_depths)
                points_proj = np.round(points_proj)

                # find valid indices
                valid = (points_proj[0,:] >= 0) & (points_proj[1,:] >= 0) & (points_proj[0,:] < self.width_) & (points_proj[1,:] < self.height_)

                return points_proj[:2,:].astype(np.int), np.where(valid)[0]

        def deproject(self, depth_image):
                '''
                Deprojects a depth image (2D numpy float array) into a point cloud

                Params:
                   depth_image: (HxW numpy array of floats) 2D depth image to project
                Returns:
                   3xN numpy float array of 3D points
                '''
                height = depth_image.shape[0]
                width = depth_image.shape[1]

                # create homogeneous pixels 
                row_indices = np.arange(height)
                col_indices = np.arange(width)
                pixel_grid = np.meshgrid(col_indices, row_indices)
                pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
                pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
                depth_arr = np.tile(depth_image.flatten(), [3,1])

                # deproject
                points_3d = depth_arr * np.linalg.inv(self.K_).dot(pixels_homog)
                return points_3d
                
