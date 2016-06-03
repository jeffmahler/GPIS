"""
Tests for the SDF class
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
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def test_sdf_surface_normals(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)
    
    # visualize finger close on everythang
    indices = [2]#, 11, 29, 102, 179]
    for ind in indices:
        grasp = grasps[ind]
        contacts_found, contacts = grasp.close_fingers(obj, vis=True)

        if not contacts_found:
            logging.info('Contacts not found')

        as_grid = obj.sdf.transform_pt_obj_to_grid(contacts[0].point)
        normal, c1_pts = obj.sdf.surface_normal(as_grid)
        as_grid = obj.sdf.transform_pt_obj_to_grid(contacts[1].point)
        normal, c2_pts = obj.sdf.surface_normal(as_grid)

        contacts[0].plot_friction_cone(color='y')
        contacts[1].plot_friction_cone(color='c')
        
        ax = plt.gca()
        ax.scatter(c1_pts[:,0], c1_pts[:,1], c1_pts[:,2], c='g', s=120)
        ax.scatter(c2_pts[:,0], c2_pts[:,1], c2_pts[:,2], c='k', s=120)
        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
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
    test_sdf_surface_normals(dataset, config)
