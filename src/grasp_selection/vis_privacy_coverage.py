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
import privacy_coverage as coverage
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

    dataset = config['hacky_vis_stuff']
    obj_names = list(config['hacky_vis_stuff'])
    for i in range(len(dataset)):
        name = dataset[i]
        obj_names[i] = name.replace('masked_hull', 'no_mask').replace('masked_bbox', 'no_mask').split('/')[-1]
        d_mesh = obj_file.ObjFile(os.path.join('masks', name+'.obj')).read()
        d_sdf = sdf_file.SdfFile(os.path.join('masks', name+'.sdf')).read()
        d_key = name.split('/')[-1]
        dataset[i] = go.GraspableObject3D(d_sdf, d_mesh, key=d_key)


    # loop through the objects and visualize coverage for each
    for obj_name, obj in zip(obj_names, dataset):

        # Stable poses
        clean_name = obj.key.replace('_no_mask', '').replace('_masked_bbox', '').replace('_masked_hull', '')
        clean_mesh = obj_file.ObjFile(os.path.join('masks', clean_name+'.obj')).read()
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(clean_name))
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)

        # Load the coverage data
        mask_filename = os.path.join('masks', '{}_mask.npy'.format(clean_name))
        grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(clean_name))
        indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(clean_name))
        quality_filename = os.path.join(config['out_dir'], '{}_qualities.npy'.format(clean_name))
        f = open(grasps_filename, 'r')
        grasps = pkl.load(f)
        index_pairs = np.load(indices_filename)
        qualities = np.load(quality_filename)
        # mask = np.load(mask_filename).tolist()
        mask = [] # TODO DELETE

        # Prune grasps that contact the mask
        public_grasps = []
        public_index_pairs = []
        public_qualities = []
        private_grasps = []
        private_index_pairs = []
        private_qualities = []
        for grasp, index_pair, quality in zip(grasps, index_pairs, qualities):
            if index_pair[0] not in mask and index_pair[1] not in mask:
                public_grasps.append(grasp)
                public_index_pairs.append(index_pair)
                public_qualities.append(quality)
            else:
                private_grasps.append(grasp)
                private_index_pairs.append(index_pair)
                private_qualities.append(quality)                

        # compute metrics
        mn, mx = clean_mesh.bounding_box()
        alpha = max(max(mx[2] - mn[2], mx[1] - mn[1]), mx[0] - mn[0])
        privacy_metric = clean_mesh.surface_area(mask) / clean_mesh.surface_area()
        coverage_metric = 0.0
        for g_j in private_grasps:
            min_dist = np.inf
            for g_i in public_grasps:
                dist = g.ParallelJawPtGrasp3D.distance(g_i, g_j, alpha=alpha)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > coverage_metric:
                coverage_metric = min_dist
        logging.info('Privacy metric: %f' %(privacy_metric))
        logging.info('Coverage metric: %f' %(coverage_metric))

        # sort
        grasps_and_qualities = zip(public_grasps, public_qualities)
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
                                most_stable_pose, vis_transform=True, max_rays=500, high=None)
            mlab.show()
            exit(0)
        except:
            pass
            #continue
