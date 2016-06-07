"""
A bunch of functions for image processing
Author: Jeff Mahler
"""
import cv2
import logging
import IPython
import matplotlib.pyplot as plt
import numpy as np
import tfx

import similarity_tf as stf

def ij_to_linear(i, j, width):
    return i + j.dot(width)

def linear_to_ij(ind, width):
    return np.c_[ind % width, ind / width]

class DepthImageProcessing:
    """ A depth image is defined as a 2-D numpy float array """  

    @staticmethod
    def threshold_depth(depth_im, front_thresh=0.0, rear_thresh=100.0):
        """ Sets all values less than |front_thresh| and greater than |rear_thresh| to 0.0 """
        depth_im[depth_im < front_thresh] = 0.0
        depth_im[depth_im > rear_thresh] = 0.0
        return depth_im

    @staticmethod
    def mask_binary(depth_im, binary_im):
        """ Sets all values of |depth_im| where |binary_im| is zero to 0.0 """
        ind = np.where(binary_im == 0)
        depth_im[ind[0], ind[1]] = 0.0
        return depth_im

    @staticmethod
    def crop_center(depth_im, width, height):
        """ Crops an image in the center to be of widthxheight """
        old_height, old_width = depth_im.shape
        if width > old_width or height > old_height:
            logging.warning('Cannot crop image with width or height greater than original image dimensions')
            return depth_im

        start_row = old_height / 2 - height / 2
        start_col = old_width / 2 - width / 2
        depth_im_crop = depth_im[start_row:start_row+height, start_col:start_col+width]
        return depth_im_crop

    @staticmethod
    def center_nonzero_depth(depth_im):
        """ Recenters the depth image on areas of nonzero depth """
        # get the center of the nonzero pixels
        nonzero_px = np.where(depth_im != 0.0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        mean_px = np.mean(nonzero_px, axis=0)
        center_px = np.array(depth_im.shape) / 2.0
        diff_px = center_px - mean_px
        height = depth_im.shape[0]
        width = depth_im.shape[1]

        # transform image
        nonzero_px_tf = nonzero_px + diff_px
        nonzero_px_tf[:,0] = np.max(np.c_[np.zeros(nonzero_px_tf[:,0].shape), nonzero_px_tf[:,0]], axis=1)
        nonzero_px_tf[:,0] = np.min(np.c_[(height-1)*np.ones(nonzero_px_tf[:,0].shape), nonzero_px_tf[:,0]], axis=1)
        nonzero_px_tf[:,1] = np.max(np.c_[np.zeros(nonzero_px_tf[:,1].shape), nonzero_px_tf[:,1]], axis=1)
        nonzero_px_tf[:,1] = np.min(np.c_[(width-1)*np.ones(nonzero_px_tf[:,1].shape), nonzero_px_tf[:,1]], axis=1)
        nonzero_px = nonzero_px.astype(np.uint16)
        nonzero_px_tf = nonzero_px_tf.astype(np.uint16)
        depth_im_tf = np.zeros(depth_im.shape)
        depth_im_tf[nonzero_px_tf[:,0], nonzero_px_tf[:,1]] = depth_im[nonzero_px[:,0], nonzero_px[:,1]]
        return depth_im_tf, diff_px

    @staticmethod
    def depth_to_binary(depth_im, threshold=0.0):
        """ Creates a binary image of all points in |depth_im| with depth greater than |threshold| """
        binary_im = 1 * (depth_im > threshold)
        binary_im = binary_im.astype(np.uint8)
        return binary_im

    @staticmethod
    def compute_normals(depth_im, camera_params):
        """ Computes normal maps for |depth_im| """
        # create a point cloud grid
        point_cloud = camera_params.deproject(depth_im)
        point_cloud_grid = point_cloud.T.reshape(depth_im.shape[0], depth_im.shape[1], 3)

        # compute normals
        normals = np.zeros([depth_im.shape[0], depth_im.shape[1], 3])
        for i in range(depth_im.shape[0]-1):
            for j in range(depth_im.shape[1]-1):
                p = point_cloud_grid[i,j,:]
                p_r = point_cloud_grid[i,j+1,:]
                p_b = point_cloud_grid[i+1,j,:]
                if np.linalg.norm(p) > 0 and np.linalg.norm(p_r) > 0 and np.linalg.norm(p_b) > 0:
                    v_r = p_r - p
                    v_r = v_r / np.linalg.norm(v_r)
                    v_b = p_b - p
                    v_b = v_b / np.linalg.norm(v_b)
                    normals[i,j,:] = np.cross(v_b, v_r)
                    normals[i,j,:] = normals[i,j,:] / np.linalg.norm(normals[i,j,:])

        # grab only the points
        normals = normals.reshape(depth_im.shape[0]*depth_im.shape[1], 3).T
        return normals, point_cloud

    @staticmethod
    def image_shift_to_transform(source_depth_im, target_depth_im, camera_params, diff_px):
        """ Converts 2D pixel shift transformation between two depth images into a 3D transformation """
        nonzero_source_depth_px = np.where(source_depth_im > 0)
        if nonzero_source_depth_px[0].shape[0] == 0:
            return stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)), from_frame='camera', to_frame='camera')

        source_px = np.array([nonzero_source_depth_px[0][0], nonzero_source_depth_px[1][0]])
        source_depth = source_depth_im[source_px[0], source_px[1]]
        target_px = source_px + diff_px
        source_point = source_depth * np.linalg.inv(camera_params.K_).dot(np.array([source_px[1], source_px[0], 1]))
        target_point = source_depth * np.linalg.inv(camera_params.K_).dot(np.array([target_px[1], target_px[0], 1]))

        translation_source_target = target_point - source_point
        translation_source_target[2] = 0
        tf_source_target = tfx.pose(np.eye(3), translation_source_target)
        return stf.SimilarityTransform3D(pose=tfx.pose(tf_source_target), from_frame='camera', to_frame='camera')

class GrayscaleImageProcessing:
    pass

class ColorImageProcessing:
    @staticmethod
    def find_chessboard(raw_image, sx=6, sy=9, vis=False):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((sx*sy,3), np.float32)
        objp[:,:2] = np.mgrid[0:sx,0:sy].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # create images
        img = raw_image.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (sx,sy), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            if vis:
                cv2.drawChessboardCorners(img_rgb, (sx,sy), corners, ret)
                cv2.imshow('img', img_rgb)
                cv2.waitKey(500)

        cv2.destroyAllWindows()
        if corners is not None:
            return corners.squeeze()
        return None


class BinaryImageProcessing:
    @staticmethod
    def prune_contours(binary_im, area_thresh=1000.0):
        """ Prunes all binary image connected components with area less than |area_thresh| """
        # get all contours (connected components) from the binary image
        contours = cv2.findContours(binary_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours[0])
        pruned_contours = []

        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[0][i])
            if area > area_thresh:
                pruned_contours.append(contours[0][i])

        # mask out bad areas in the image
        binary_im_ch = np.zeros([binary_im.shape[0], binary_im.shape[1], 3])
        for contour in pruned_contours:
            cv2.fillPoly(binary_im_ch, pts=[contour], color=(255,255,255))
        binary_im_ch = binary_im_ch[:,:,0] # convert back to one channel
        return binary_im_ch

    @staticmethod
    def binary_to_grayscale(binary_im):
        """ Converts a binary image to grayscale """
        grayscale_im = np.zeros([binary_im.shape[0], binary_im.shape[1], 3])
        grayscale_im[:,:,0] = 255.0 * binary_im
        grayscale_im[:,:,1] = 255.0 * binary_im
        grayscale_im[:,:,2] = 255.0 * binary_im
        return grayscale_im.astype(np.uint8)

class PointCloudProcessing:
    """ A point cloud is defined as a 3xN numpy float array, where N is the number of points """  

    @staticmethod
    def fit_plane(point_cloud):
        """ Fits a plane to the point cloud """
        X = np.c_[point_cloud[:2,:].T, np.ones(point_cloud.shape[1])]
        y = point_cloud[2,:].T
        A = X.T.dot(X)
        b = X.T.dot(y)
        w = np.linalg.inv(A).dot(b)
        n = np.array([w[0], w[1], -1])
        n = n / np.linalg.norm(n)
        x0 = np.mean(point_cloud, axis=1)
        x0 = np.reshape(x0, [3, 1])
        return n, x0

    @staticmethod
    def prune_points_above_plane(point_cloud, n, x0):
        """ Removes the points above the plane defined by n and x0 """
        points_of_interest = (point_cloud - np.tile(x0, [1, point_cloud.shape[1]])).T.dot(n) > 0
        points_of_interest = (point_cloud[2,:] > 0) & points_of_interest
        points_of_interest = np.where(points_of_interest)[0]

        pruned_indices = np.setdiff1d(np.arange(point_cloud.shape[1]), points_of_interest)
        point_cloud_pruned = point_cloud[:, points_of_interest]
        return point_cloud_pruned, pruned_indices

    @staticmethod
    def remove_zero_points(point_cloud):
        """ Removes points of zero depth """
        points_of_interest = np.where(point_cloud[2,:] != 0.0)[0]
        point_cloud = point_cloud[:, points_of_interest]
        return point_cloud

class NormalCloudProcessing:
    @staticmethod
    def remove_zero_normals(normal_cloud):
        """ Removes points of zero depth """
        points_of_interest = np.where(np.linalg.norm(normal_cloud, axis=0) != 0.0)[0]
        normal_cloud = normal_cloud[:, points_of_interest]
        return normal_cloud
    
