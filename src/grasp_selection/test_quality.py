"""
Tests for grasp quality
Author: Jeff Mahler
"""
import argparse
import copy
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_collision_checker as gcc
import grasp_sampler as gs
import gripper as gr
import json_serialization as jsons
import kernels
import models
import mayavi_visualizer as mv
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def test_ferrari_canny_L1(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)
    
    # visualize finger close on everythang
    vis_indices = []#[58]
    fc = []
    frcl = []
    for i, grasp in enumerate(grasps):
        eps = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'ferrari_canny_L1', friction_coef=config['friction_coef'],
                                                       num_cone_faces=config['num_cone_faces'], soft_fingers=True)
        fcl = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'force_closure', friction_coef=config['friction_coef'],
                                                       num_cone_faces=config['num_cone_faces'], soft_fingers=True)
        fc.append(eps)
        frcl.append(fcl)

        logging.info('Grasp %d ferrari canny = %.5f, force closure = %d' %(grasp.grasp_id, eps, fcl))

        if grasp.grasp_id in vis_indices:
            contacts_found, contacts = grasp.close_fingers(obj, vis=True)
            if not contacts_found:
                logging.info('Contacts not found')

            contacts[0].plot_friction_cone(color='y')
            contacts[1].plot_friction_cone(color='c')
        
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

    gripper = gr.RobotGripper.load(gripper_name)
    grasps_and_qualities = zip(grasps, fc)
    grasps_and_qualities.sort(key = lambda x: x[1], reverse=True)
    for grasp, fc in grasps_and_qualities:
        logging.info('Grasp %d quality %f' %(grasp.grasp_id, fc))
        mlab.figure()
        mv.MayaviVisualizer.plot_mesh(obj.mesh)
        mv.MayaviVisualizer.plot_gripper(grasp, gripper=gripper)
        mlab.show()

    num_bins = 25
    font_size = 15
    i = 0
    for eps, force in zip(fc, frcl):
        if eps > 0 and force == 0:
            logging.info('Ferrari canny positive for out of fc at %d' %(i))
        elif eps == 0 and force == 1:
            logging.info('Ferrari canny zero for fc at %d' %(i))
        i += 1

    grasp_success_hist, grasp_success_bins = np.histogram(fc, bins=num_bins, range=(0,np.max(fc)))
    width = (grasp_success_bins[1] - grasp_success_bins[0])
    
    plt.figure()
    plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
    plt.title('Histogram', fontsize=font_size)
    plt.xlabel('Ferrari Canny', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    config_filename = 'cfg/test/test_sdf.yaml'

    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_ferrari_canny_L1(dataset, config)
