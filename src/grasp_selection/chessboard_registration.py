import IPython
import logging
import math
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import camera_params as cp
import experiment_config as ec
import image_processing as ip
import rgbd_sensor as rs
import tfx

if __name__ == '__main__':

    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # check that a yaml configuration file is specified
    if len(sys.argv) != 2 or len(sys.argv[1]) < 5 or sys.argv[1][len(sys.argv[1])-5:] != '.yaml':
        logging.error('No yaml configuration file specified')
        exit(-1)

    # get camera sensor object
    s = rs.RgbdSensor()

    #reading configuration file
    #os.chdir(os.getcwd() + '/../../')
    config = ec.ExperimentConfig(sys.argv[1])

    # chessboard dimensions
    sx = config['sx']
    sy = config['sy']
    load = config['load']
    chessboard_thickness = config['chessboard_thickness']

    # repeat registration multiple times and average results
    R = np.zeros([3,3])
    t = np.zeros([3,1])
    points_3d_plane = np.zeros([3, sx*sy])
    for k in range(config['num_avg']):
        if load:
            f = open('data/test/rgbd/depth_im.npy', 'r')
            depth_im = np.load(f)
            f = open('data/test/rgbd/color_im.npy', 'r')
            color_im = np.load(f)
            f = open('data/test/rgbd/corners.npy', 'r')
            corner_px = np.load(f)
        else:
            # average a bunch of depth images together
            num_images = config['num_images']
            depth_im = np.zeros([s.height_, s.width_])
            counts = np.zeros([s.height_, s.width_])
            for i in range(num_images):
                new_depth_im = s.get_depth_image()

                depth_im = depth_im + new_depth_im
                counts = counts + np.array(new_depth_im > 0.0)

            depth_im[depth_im > 0] = depth_im[depth_im > 0] / counts[depth_im > 0]
            color_im = s.get_color_image()
            corner_px = ip.ColorImageProcessing.find_chessboard(color_im, sx=config['sx'], sy=config['sy'], vis=True)

        depth_im[depth_im > 1.0] = 0.0

        if corner_px is None:
            logging.error('No chessboard detected')
            exit(-1)

        # project points into 3D
        camera_params = cp.CameraParams(s.height_, s.width_, 525.)
        points_3d = camera_params.deproject(depth_im)

        # get round chessboard ind
        corner_px_round = np.round(corner_px).astype(np.uint16)
        corner_ind = ip.ij_to_linear(corner_px_round[:,0], corner_px_round[:,1], s.width_)

        # average 3d points
        points_3d_plane = (k * points_3d_plane + points_3d[:, corner_ind]) / (k + 1)
        print('Iter: %d' %(k))

    # fit a plane to the chessboard corners
    X = np.c_[points_3d_plane[:2,:].T, np.ones(points_3d_plane.shape[1])]
    y = points_3d_plane[2,:].T
    A = X.T.dot(X)
    b = X.T.dot(y)
    w = np.linalg.inv(A).dot(b)
    n = np.array([w[0], w[1], -1])
    n = n / np.linalg.norm(n)
    mean_point_plane = np.mean(points_3d_plane, axis=1)
    mean_point_plane = mean_point_plane - chessboard_thickness * n
    mean_point_plane = np.reshape(mean_point_plane, [3, 1])

    # find x-axis of the chessboard coordinates on the fitted plane
    dir_3d = points_3d_plane - np.array([mean_point_plane[:,0] for i in range(points_3d_plane.shape[1])]).T
    orientation = np.sum(dir_3d[:,:int(math.floor(config['sx']*(config['sy']-1)/2.0))], axis=1)
    orientation = orientation - np.sum(dir_3d[:,int(math.ceil(config['sx']*(config['sy']+1)/2.0)):], axis=1)
    proj_orient = orientation - np.vdot(orientation, n)*n
    proj_orient = proj_orient / np.linalg.norm(proj_orient)

    # determine y-axis from z-axis (normal vector) and x-axis
    yaxis = np.cross(n, proj_orient)

    if yaxis[0] < 0:
        proj_orient = -proj_orient
        yaxis = -yaxis

    # produce translation and rotation from plane center and chessboard basis
    rotation = np.hstack((proj_orient[:,np.newaxis], yaxis[:,np.newaxis], n[:,np.newaxis])).T
    translation = mean_point_plane

    # save tranformation arrays
    theta = config['table_angle'] * np.pi / 180.0
    rotation_camera_cb = rotation
    rotation_world_table = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
    rotation_table_cb = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])
    rotation_camera_world = rotation_camera_cb.T.dot(rotation_world_table).dot(rotation_table_cb)

    print('Final Result')
    print('Rotation: ')
    print(rotation_camera_world)
    print('Translation: ')
    print(translation)

    f = open(config['save_dir']+'rotation_camera_cb.npy', 'w')
    np.save(f, rotation_camera_world)
    f = open(config['save_dir']+'translation_camera_cb.npy', 'w')
    np.save(f, translation)
    f = open(config['save_dir']+'corners_cb.npy', 'w')
    np.save(f, points_3d_plane)

    ###########################
    ##### Plotting things #####
    ###########################
    if config['debug']:
        # plot and display corner detection on camera image
        plt.scatter(corner_px_round[:,0], corner_px_round[:,1])
        plt.show()

        # separate vector into components for easier plotting
        mpx, mpy, mpz = translation

        # axis vectors scaled down for better visibility
        xdraw = proj_orient / config['scale_amt']
        ydraw =  yaxis / config['scale_amt']
        zdraw = n / config['scale_amt']

        # plot chessboard corners in camera basis
        ax = Axes3D(plt.figure())
        ax.scatter(points_3d_plane[0,:], points_3d_plane[1,:], points_3d_plane[2,:])

        # plot axis vectors
        t = translation
        ax.plot([t[0],t[0]+xdraw[0]], [t[1],t[1]+xdraw[1]], zs=[t[2],t[2]+xdraw[2]], color='r')
        ax.plot([t[0],t[0]+ydraw[0]], [t[1],t[1]+ydraw[1]], zs=[t[2],t[2]+ydraw[2]], color='g')
        ax.plot([t[0],t[0]+zdraw[0]], [t[1],t[1]+zdraw[1]], zs=[t[2],t[2]+zdraw[2]], color='b')

        # display point plot in camera perspective
        plt.show()

        # rotate axis vectors to new basis
        xdraw = np.dot(rotation, proj_orient) / config['scale_amt'] 
        ydraw = np.dot(rotation, yaxis) / config['scale_amt']
        zdraw = np.dot(rotation, n) / config['scale_amt']

        # plot chessboard corners in world basis
        translate_matrix = np.hstack(tuple([translation for i in range(points_3d_plane.shape[1])]))
        transformed_points = np.dot(rotation, points_3d_plane - translate_matrix)
        ax = Axes3D(plt.figure())
        ax.scatter(transformed_points[0,:], transformed_points[1,:], transformed_points[2,:])

        # plot axis vectors
        ax.plot([0,xdraw[0]], [0,xdraw[1]], zs=[0,xdraw[2]], color='r')
        ax.plot([0,ydraw[0]], [0,ydraw[1]], zs=[0,ydraw[2]], color='g')
        ax.plot([0,zdraw[0]], [0,zdraw[1]], zs=[0,zdraw[2]], color='b')

        # plot camera position in world basis
        camera = np.dot(rotation, -translation)
        ax.scatter(camera[0,0], camera[1,0], camera[2,0], color='y')

        # display 3D point plot in world coordinates
        ax.set_xlim(-5.0/config['scale_amt'],5.0/config['scale_amt'])
        ax.set_ylim(-5.0/config['scale_amt'],5.0/config['scale_amt'])
        ax.set_zlim(-5.0/config['scale_amt'],5.0/config['scale_amt'])
        plt.show()
