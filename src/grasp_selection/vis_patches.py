"""
Script to visualize object patches
Author: Jeff Mahler
"""
import argparse
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mv
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_sampler as gs
import json_serialization as jsons
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def pw_filenames(data_dir):
    """ Load in projection window filenames """
    w1_tag = 'w1'
    w2_tag = 'w2'
    pw_tag = 'projection_window'
    po_tag = 'patch_orientation'
    tan_tag = 'tangents'
    contact_tag = 'moment_arms'
    quality_tag = 'pfc_f_0.2'
    pw_files = []
    for filename in os.listdir(data_dir):
        if filename.find(pw_tag) != -1 or filename.find(po_tag) != -1 or filename.find(contact_tag) != -1 \
                or filename.find(tan_tag) != -1 or filename.find(quality_tag) != -1:
            pw_files.append(filename)
    pw_files.sort(key=lambda x: int(x[-6:-4]))

    pw1_files = []
    pw2_files = []
    po_files = []
    contact_files = []
    tangents_files = []
    quality_files = []
    for filename in pw_files:
        if filename.find(w1_tag) != -1:
            pw1_files.append(filename)
        elif filename.find(w2_tag) != -1:
            pw2_files.append(filename)
        elif filename.find(po_tag) != -1:
            po_files.append(filename)
        elif filename.find(tan_tag) != -1:
            tangents_files.append(filename)
        elif filename.find(contact_tag) != -1:
            contact_files.append(filename)
        elif filename.find(quality_tag) != -1:
            quality_files.append(filename)
    return pw1_files, pw2_files, po_files, tangents_files, contact_files, quality_files

def plot_grasp(proj_window1, proj_window2, contacts, patch_orientation, quality, res=0.00416):
    d = int(np.sqrt(proj_window1.shape[0]))
    w = d / 2
    proj_window1 = proj_window1.reshape(d,d)
    proj_window2 = proj_window2.reshape(d,d)
    grad_x_window1, grad_y_window1 = np.gradient(proj_window1)
    grad_x_window2, grad_y_window2 = np.gradient(proj_window2)
    contact1 = contacts[:3]
    contact2 = contacts[3:]
    po1 = patch_orientation
    po1 = po1 / np.linalg.norm(po1)
    po2 = -patch_orientation

    if np.abs(proj_window1[6,6]) > 0.01 and np.abs(proj_window2[6,6]) > 0.01:
        logging.info('Invalid. Skipping plot')
        return

    render_width = 0.05
    offset1 = proj_window1[6,6]
    offset2 = proj_window2[6,6]    

    center = 0.5 * (contact1 + contact2)
    contact1 = center + 0.5 * render_width * po1
    contact2 = center + 0.5 * render_width * po2

    U, _, _ = np.linalg.svd(po1.reshape(3,1))
    tangents_x1 = U[:,2]
    tangents_y1 = U[:,1]
    U, _, _ = np.linalg.svd(po2.reshape(3,1))
    tangents_x2 = U[:,2]
    tangents_y2 = U[:,1]

    R_world_patch1 = np.array([tangents_x1, tangents_y1, po1]).T
    t_world_patch1 = contact1
    T_world_patch1 = stf.SimilarityTransform3D(pose=tfx.pose(R_world_patch1, t_world_patch1),
                                               from_frame='patch1', to_frame='world')

    R_world_patch2 = np.array([tangents_x2, -tangents_y2, po2]).T
    t_world_patch2 = contact2
    T_world_patch2 = stf.SimilarityTransform3D(pose=tfx.pose(R_world_patch2, t_world_patch2),
                                               from_frame='patch2', to_frame='world')

    dist_thresh = 0.05
    grad_thresh = 0.01
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    points_3d_w1 = np.zeros([0,3])
    points_3d_w2 = np.zeros([0,3])
    for y in range(-w,w+1):
        for x in range(-w,w+1):
            x_3d = x*res
            y_3d = y*res
            if np.abs(proj_window1[y+w, x+w]-offset1) < dist_thresh and \
                    np.abs(grad_x_window1[y+w, x+w]) < grad_thresh and \
                    np.abs(grad_y_window1[y+w, x+w]) < grad_thresh:
                x1_list.append(x)
                y1_list.append(y)
                points_3d_w1 = np.r_[points_3d_w1,
                                     np.array([[x_3d, y_3d, -proj_window1[y+w, x+w] + offset1]])]
            if np.abs(proj_window2[y+w, x+w]-offset2) < dist_thresh and \
                    np.abs(grad_x_window2[y+w, x+w]) < grad_thresh and \
                    np.abs(grad_y_window2[y+w, x+w]) < grad_thresh:
                x2_list.append(x)
                y2_list.append(y)
                points_3d_w2 = np.r_[points_3d_w2,
                                     np.array([[x_3d, -y_3d, -proj_window2[y+w, x+w] + offset2]])]

                
    if len(x1_list) > 3 and len(x2_list) > 3 and len(y1_list) > 3 and len(y2_list) > 3:
        tri1 = mtri.Triangulation(x1_list, y1_list)
        tri2 = mtri.Triangulation(x2_list, y2_list)
        
        points_3d_w1 = T_world_patch1.apply(points_3d_w1.T).T
        points_3d_w2 = T_world_patch2.apply(points_3d_w2.T).T
        
        axis = np.array([contact1, contact2])

        mv.figure()
        mv.triangular_mesh(points_3d_w1[:,0], points_3d_w1[:,1], points_3d_w1[:,2], tri1.triangles,
                           representation='surface', color=(1,1,0))
        mv.triangular_mesh(points_3d_w1[:,0], points_3d_w1[:,1], points_3d_w1[:,2], tri1.triangles,
                           representation='wireframe', color=(0,0,0))
        mv.points3d(contact1[0], contact1[1], contact1[2], color=(1,0,0), scale_factor=0.005)
        mv.triangular_mesh(points_3d_w2[:,0], points_3d_w2[:,1], points_3d_w2[:,2], tri2.triangles,
                           representation='surface', color=(1,1,0))
        mv.triangular_mesh(points_3d_w2[:,0], points_3d_w2[:,1], points_3d_w2[:,2], tri2.triangles,
                           representation='wireframe', color=(0,0,0))
        mv.points3d(contact2[0], contact2[1], contact2[2], color=(1,0,0), scale_factor=0.005)
        mv.plot3d(axis[:,0], axis[:,1], axis[:,2], tube_radius=0.0005, color=(0,0,1))
        mv.title('Robustness = %.4f' %(quality))
        mv.show()
        
        print(np.max(grad_x_window1))
        print(np.max(grad_y_window1))
        print(np.max(grad_x_window2))
        print(np.max(grad_y_window2))

if __name__ == '__main__':
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('input_dir')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # load in projection window filenames
    pw1_filenames, pw2_filenames, po_filenames, tangents_filenames, contact_filenames, quality_filenames = \
        pw_filenames(args.input_dir)

    # plot windows
    j = 0
    for pw1_filename, pw2_filename, po_filename, contact_filename, quality_filename in \
            zip(pw1_filenames, pw2_filenames, po_filenames, contact_filenames, quality_filenames):

        pw1_arr = np.load(os.path.join(args.input_dir, pw1_filename))['arr_0']
        pw2_arr = np.load(os.path.join(args.input_dir, pw2_filename))['arr_0']
        po_arr = np.load(os.path.join(args.input_dir, po_filename))['arr_0']
        contact_arr = np.load(os.path.join(args.input_dir, contact_filename))['arr_0']
        quality_arr = np.load(os.path.join(args.input_dir, quality_filename))['arr_0']
        
        ind = np.where((quality_arr > 0.9) & (quality_arr < 1.0))[0]
        np.random.shuffle(ind)

        num_windows = pw1_arr.shape[0]
        for i in ind.tolist():#range(num_windows):
            logging.info('Displaying patch %d' %(i))
            plot_grasp(pw1_arr[i,:], pw2_arr[i,:], contact_arr[i,:], po_arr[i,:], quality_arr[i])
        j = j+1
