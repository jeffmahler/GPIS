"""
Script to visualize the computed coverage results
Author: Jeff Mahler
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython

from mayavi import mlab

import os
import json
import logging
import pickle as pkl
import random
import shutil
import sys
import time

import similarity_tf as stf
import tfx

import contacts
import privacy_coverage as coverage
import database as db
import experiment_config as ec
import gripper as gr
import grasp as g
import graspable_object as go
import mayavi_visualizer as mv
import obj_file
import quality as q
import sdf_file
import similarity_tf as stf
import stp_file as stp
import random

masked_object_tags = ['_no_mask']

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def compute_coverage_metric(grasp_universe, grasp_set, alpha=1.0):
    """ Compute the coverage metric for a grasp set compared to a discrete approximation of the grasp universe """
    # always infinite when the grasp set is zero - won't be plotted by the way
    if len(grasp_set) == 0:
        return np.inf

    coverage_metric = 0.0
    for g_j in grasp_universe:
        min_dist = np.inf
        for g_i in grasp_set:
            dist = g.ParallelJawPtGrasp3D.distance(g_i, g_j, alpha=alpha)
            if dist < min_dist:
                min_dist = dist
        if min_dist > coverage_metric:
            coverage_metric = min_dist
    return coverage_metric

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    # load the models
    data_dir = config['data_dir']
    object_keys = coverage.read_index(data_dir)
    objects = []
    masks = []
    masked_objects = []

    # setup experiment output dir
    experiment_id = gen_experiment_id()
    output_dir = config['out_dir']
    experiment_dir = os.path.join(output_dir, 'collisions-%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    _, config_root = os.path.split(config_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_root))
    logging.info('Saving analysis to %s' %(experiment_dir))

    # get the mode
    gripper = gr.RobotGripper.load(config['gripper'])
    mode = config['contact_point_mode']
    logging.info('Using contact point mode %d' %(mode))

    # read plotting params
    dpi = config['dpi']
    font_size = config['font_size']
    line_width = config['line_width']
    num_views = config['num_views']
    delta_view = 360.0 / num_views
    elevs = [55.0, 30.0]
    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))

    # load all of the objects
    for object_key in object_keys:
        logging.info('Loading object %s' %(object_key))
        
        # first load the clean mesh
        subdiv = False
        mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+db.OBJ_EXT)).read()
        mesh, tri_ind_mapping = mesh.subdivide(min_triangle_length=config['min_tri_length'])

        sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+db.SDF_EXT)).read()
        objects.append(go.GraspableObject3D(sdf, mesh, key=object_key))
        objects[-1].model_name_ = os.path.join(data_dir, object_key+db.OBJ_EXT)
        masks.append(np.load(os.path.join(data_dir, object_key+'_mask.npy')))

        new_mask = []
        mask = masks[-1]
        for index in mask.tolist():
            for new_index in tri_ind_mapping[index]:
                new_mask.append(new_index)
        masks[-1] = np.array(new_mask)

        masked_objects.append([])

        # then load the masked versions
        for mask_tag in masked_object_tags:
            mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)).read()
            mesh, tri_ind_mapping = mesh.subdivide(min_triangle_length=config['min_tri_length'])
            sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+mask_tag+db.SDF_EXT)).read()
            masked_objects[-1].append(go.GraspableObject3D(sdf, mesh, key=object_key+mask_tag))
            masked_objects[-1][-1].model_name_ = os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)

    # loop through the objects and compute coverage for each
    privacy_metrics = {}
    coverage_metrics = {}
    for graspable, masked_graspables, mask in zip(objects, masked_objects, masks):
        obj_name = graspable.key
        logging.info('Analyzing coverage for object %s' %(obj_name))

        # compute stable poses (for later down the pipeline, maybe we should remove this...)
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=config['min_prob'])
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)

        # visualize grasps with stable pose
        stable_pose_probs = np.array([s.p for s in stable_poses])
        highest_index = np.where(stable_pose_probs == np.max(stable_pose_probs))[0]
        most_stable_pose = stable_poses[0]#highest_index[0]]
        if graspable.key.find('endstop') != -1:
            most_stable_pose = stable_poses[-1]#highest_index[0]]

        # load the coverage data
        logging.info('Loading grasps')
        grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
        logging.info('Loading indices')
        indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
        logging.info('Loading quality')
        quality_filename = os.path.join(config['out_dir'], '{}_qualities.pkl'.format(obj_name))
        f = open(grasps_filename, 'r')
        grasps = pkl.load(f)
        for i, grasp in enumerate(grasps):
            grasp.grasp_id_ = i
        index_pairs = np.load(indices_filename)
        f = open(quality_filename, 'r')
        qualities = pkl.load(f)

        # take a subset of the grasps, indices, and qualities
        num_grasps_to_analyze = config['num_grasps_analyze']
        grasps_indices_qualities = zip(grasps, index_pairs, qualities)
        random.shuffle(grasps_indices_qualities)
        grasps_indices_qualities = grasps_indices_qualities[:num_grasps_to_analyze]
        grasps = [giq[0] for giq in grasps_indices_qualities]
        index_pairs = [giq[1] for giq in grasps_indices_qualities]
        qualities = [giq[2] for giq in grasps_indices_qualities]

        # get robust grasp qualties
        robust_qualities = []
        for quality in qualities:
            robust_qualities.append(quality['robust_force_closure'])

        # split covering set into those that contact the mask and those that don't
        public_grasps = []
        public_index_pairs = []
        public_qualities = []
        for grasp, index_pair, quality in zip(grasps, index_pairs, robust_qualities):
            if index_pair[0] not in mask and index_pair[1] not in mask:
                grasp.grasp_id_ = len(public_grasps)
                public_grasps.append(grasp)
                public_index_pairs.append(index_pair)
                public_qualities.append(quality)

        # plotting
        public_mask = []
        for i in range(len(graspable.mesh.triangles())):
            if i not in mask.tolist():
                public_mask.append(i)
        private_mesh = graspable.mesh.mask(mask)
        private_mesh_cvh = private_mesh.convex_hull()
        private_mesh_cvh.rescale(1.025, center=True) # rescaling for weird triangle flicker (which could be fixed with fancy zippering later)
        public_mesh = graspable.mesh.mask(np.array(public_mask))

        mlab.clf()
        T_obj_world = mv.MayaviVisualizer.plot_stable_pose(graspable.mesh, most_stable_pose, d=0.15, color=(0.6,0.6,0.6))        
        for i, elev in enumerate(elevs):
            for j in range(num_views):
                az = j * delta_view
                mlab.view(azimuth=az, elevation=elev, distance=config['cam_distance'], focalpoint=T_obj_world.inverse().translation)
                figname = 'table_%s_view_%d.png' %(obj_name, j)
                mlab.savefig(os.path.join(experiment_dir, figname))
        exit(0)

        # analyze collision free grasps
        logging.info('Analyzing collision free grasps for original object')
        public_coll_free_grasps, public_coll_free_qualities = \
            coverage.prune_grasps_in_collision(graspable, public_grasps, public_qualities,
                                               gripper, most_stable_pose, config)


        mlab.clf()
        coverage.vis_coverage(public_mesh, [private_mesh], public_coll_free_grasps, public_coll_free_qualities,
                              most_stable_pose, config, gripper=None, color_gripper=False, color_grasps=False, plot_table=False,
                              max_display=config['max_display'], rank=False)
        for i, elev in enumerate(elevs):
            for j in range(num_views):
                az = j * delta_view
                mlab.view(azimuth=az, elevation=elev, distance=config['cam_distance'], focalpoint=T_obj_world.inverse().translation)
                figname = 'grasps_%s_gripper_view_%d.png' %(obj_name, j)
                mlab.savefig(os.path.join(experiment_dir, figname))

        mlab.clf()
        coverage.vis_coverage(public_mesh, [private_mesh_cvh], public_coll_free_grasps, public_coll_free_qualities,
                              most_stable_pose, config, gripper=None, color_gripper=False, color_grasps=True,
                              max_display=config['max_display'], rank=False)
        for i, elev in enumerate(elevs):
            for j in range(num_views):
                az = j * delta_view
                mlab.view(azimuth=az, elevation=elev, distance=config['cam_distance'], focalpoint=T_obj_world.inverse().translation)
                figname = 'grasps_colored_%s_gripper_view_%d.png' %(obj_name, j)
                mlab.savefig(os.path.join(experiment_dir, figname))

        i = 0
        for grasp, quality in zip(public_coll_free_grasps, public_coll_free_qualities):
            mlab.clf()
            coverage.vis_coverage(public_mesh, [private_mesh], [grasp], [quality],
                                  most_stable_pose, config, gripper=gripper, color_gripper=False, color_grasps=False, plot_table=False,
                                  max_display=config['max_display'], rank=False)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(azimuth=az, elevation=elevs[0], distance=config['cam_distance'], focalpoint=T_obj_world.inverse().translation)
                figname = 'grasp_%d_%s_gripper_view_%d.png' %(i, obj_name, j)
                mlab.savefig(os.path.join(experiment_dir, figname))
            i += 1


