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

GRAVITY_ACCEL = 9.81

def test_isotropic_graspable(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, config)

    mlab.figure()
    graspable_rv.obj_.mesh.visualize(color=(1,0,0))
    for sample in graspable_rv.prealloc_samples_:
        sample.mesh.visualize(color=(0,1,0), style='wireframe')
    mlab.show()

def test_stp_graspable(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    stp_id = 'pose_6'
    obj = dataset[obj_name]

    # uncertainty in the table plane
    physical_u_config = config['physical_uncertainty']
    Sigma_g_t = np.diag([physical_u_config['sigma_gripper_x'], physical_u_config['sigma_gripper_y'], physical_u_config['sigma_gripper_z']])
    Sigma_g_r = np.diag([physical_u_config['sigma_gripper_rot_x'], physical_u_config['sigma_gripper_rot_y'], physical_u_config['sigma_gripper_rot_z']])
    Sigma_o_t = np.diag([physical_u_config['sigma_obj_x'], physical_u_config['sigma_obj_y'], physical_u_config['sigma_obj_z']])
    Sigma_o_r = np.diag([physical_u_config['sigma_obj_rot_x'], physical_u_config['sigma_obj_rot_y'], physical_u_config['sigma_obj_rot_z']])
    Sigma_o_s = physical_u_config['sigma_scale']

    u_config = copy.deepcopy(config)
    u_config['sigma_rot_grasp'] = Sigma_g_r
    u_config['sigma_trans_grasp'] = Sigma_g_t
    u_config['sigma_rot_obj'] = Sigma_o_r
    u_config['sigma_trans_obj'] = Sigma_o_t
    u_config['sigma_scale_obj'] = Sigma_o_s
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, u_config)

    mlab.figure()
    graspable_rv.obj_.mesh.visualize(color=(1,0,0))
    for sample in graspable_rv.prealloc_samples_:
        sample.mesh.visualize(color=(0,1,0), style='wireframe')
    mlab.show()

def test_isotropic_grasp(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)
    grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasps[0], config)

    mlab.figure()
    obj.mesh.visualize(color=(1,0,0), style='wireframe')
    for grasp_sample in grasp_rv.prealloc_samples_:
        mv.MayaviVisualizer.plot_grasp(grasp_sample)
    mlab.show()

def test_stp_grasp(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    stp_id = 'pose_6'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)

    # uncertainty in the table plane
    physical_u_config = config['physical_uncertainty']
    Sigma_g_t = np.diag([physical_u_config['sigma_gripper_x'], physical_u_config['sigma_gripper_y'], physical_u_config['sigma_gripper_z']])
    Sigma_g_r = np.diag([physical_u_config['sigma_gripper_rot_x'], physical_u_config['sigma_gripper_rot_y'], physical_u_config['sigma_gripper_rot_z']])
    Sigma_o_t = np.diag([physical_u_config['sigma_obj_x'], physical_u_config['sigma_obj_y'], physical_u_config['sigma_obj_z']])
    Sigma_o_r = np.diag([physical_u_config['sigma_obj_rot_x'], physical_u_config['sigma_obj_rot_y'], physical_u_config['sigma_obj_rot_z']])
    Sigma_o_s = physical_u_config['sigma_scale']

    u_config = copy.deepcopy(config)
    u_config['sigma_rot_grasp'] = Sigma_g_r
    u_config['sigma_trans_grasp'] = Sigma_g_t
    u_config['sigma_rot_obj'] = Sigma_o_r
    u_config['sigma_trans_obj'] = Sigma_o_t
    u_config['sigma_scale_obj'] = Sigma_o_s
    grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasps[0], u_config)

    mlab.figure()
    obj.mesh.visualize(color=(1,0,0), style='wireframe')
    for grasp_sample in grasp_rv.prealloc_samples_:
        mv.MayaviVisualizer.plot_grasp(grasp_sample)
    mlab.show()
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_random_variables.yaml'

    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_isotropic_graspable(dataset, config)
    test_stp_graspable(dataset, config)
    test_isotropic_grasp(dataset, config)
    test_stp_grasp(dataset, config)
