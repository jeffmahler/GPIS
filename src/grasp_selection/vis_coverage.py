"""
Script to visualize the computed coverage results
Author: Jeff Mahler
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import IPython

from mayavi import mlab

import os
import logging
import pickle as pkl
import random
import sys
import time

import similarity_tf as stf
import tfx

import contacts
import coverage
import database as db
import experiment_config as ec
import grasp as g
import graspable_object as go
import obj_file
import quality as q
import sdf_file

import stp_file as stp
import random

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    dataset = database[config['datasets'].keys()[0]]

    #obj = dataset['Tube-Pipe-Fittings-Gaskets-54']#Tube-Pipe-Fittings-YBends-ReducedYBends-89']
    object_keys = dataset.object_keys
    random.shuffle(object_keys)
    #if True:#for obj in dataset:
    #   obj_name = obj.key
    for obj_name in object_keys:
        obj = dataset[obj_name]

        # Stable poses
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(obj.mesh, stable_pose_filename, min_prob=0.01)
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)
        
        # Load the coveragehe data
        grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
        indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
        quality_filename = os.path.join(config['out_dir'], '{}_qualities.npy'.format(obj_name))
        f = open(grasps_filename, 'r')
        grasps = pkl.load(f)
        index_pairs = np.load(indices_filename)
        qualities = np.load(quality_filename)

        # sort
        grasps_and_qualities = zip(grasps, qualities)
        grasps_and_qualities.sort(key = lambda x: x[1], reverse = True)
        grasps = [g[0] for g in grasps_and_qualities]
        qualities = [g[1] for g in grasps_and_qualities]

        for grasp in grasps:
            grasp.grasp_width_ = 0.15

        # Visualize grasps with stable pose
        num_grasps = min(250, len(grasps))
        stable_pose_probs = np.array([s.p for s in stable_poses])
        highest_index = np.where(stable_pose_probs == np.max(stable_pose_probs))[0]
        most_stable_pose = stable_poses[highest_index[0]]
        logging.info('About to plot rays with most stable pose.')
        try:
            coverage.vis_stable(obj, grasps[:num_grasps], qualities[:num_grasps],
                                most_stable_pose, vis_transform=True, max_rays=500)
            mlab.show()
        except:
            pass
            #continue
