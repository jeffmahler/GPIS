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
import glob

masked_object_tags = ['_no_mask', '_masked_bbox', '_masked_hull']

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def compute_coverage_metric(grasp_universe, grasp_set, alpha=1.0):
    """ Compute the coverage metric for a grasp set compared to a discrete approximation of the grasp universe """
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

    #To be: a LIST OF LISTS.
    all_progressive_masks = []

    masked_objects = []

    # setup experiment output dir
    experiment_id = gen_experiment_id()
    output_dir = config['out_dir']
    experiment_dir = os.path.join(output_dir, 'analysis-%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    logging.info('Saving analysis to %s' %(experiment_dir))

    # get the mode
    gripper = gr.RobotGripper.load(config['gripper'])
    mode = config['contact_point_mode']
    logging.info('Using contact point mode %d' %(mode))

    # load all of the objects
    object_index = 0

    for object_key in object_keys:
        
        logging.info('Loading object %s' %(object_key))
        
        # first load the clean mesh
        subdiv = False
        mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+db.OBJ_EXT)).read()
        num_tris = len(mesh.triangles())
        if num_tris < config['mesh_subdiv_threshold']:
            logging.info('Subdividing mesh')
            subdiv = True
            mesh = mesh.subdivide()

        sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+db.SDF_EXT)).read()
        objects.append(go.GraspableObject3D(sdf, mesh, key=object_key))
        objects[-1].model_name_ = os.path.join(data_dir, object_key+db.OBJ_EXT)

        #MASK LOADING
        original_object_masks = []
        progressive_object_masks = []

        for partial_mask_file in glob.glob(os.path.join(data_dir, object_key) + '_partial_mask*')
            current_partial_mask = np.load(partial_mask_file)
            
            if subdiv:
                sub_divided_mask = []
                for index in current_partial_mask.tolist():
                    sub_divided_mask.append(3*index)
                    sub_divided_mask.append(3*index+1)
                    sub_divided_mask.append(3*index+2)
                current_partial_mask = np.array(sub_divided_mask)

            progressive_mask = current_partial_mask
            for original_mask in original_object_masks:
                progressive_mask = np.append(original_mask, progressive_mask)

            original_object_masks.append(current_partial_mask)
            progressive_object_masks.append(progressive_mask)

        all_progressive_masks[object_index] = progressive_object_masks
        object_index += 1


        masked_objects.append([])
    
        # then load the masked versions
        for mask_tag in masked_object_tags:
            mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)).read()
            if subdiv:
                mesh = mesh.subdivide()
            sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+mask_tag+db.SDF_EXT)).read()
            masked_objects[-1].append(go.GraspableObject3D(sdf, mesh, key=object_key+mask_tag))
            masked_objects[-1][-1].model_name_ = os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)


    # loop through the objects and compute coverage for each
    privacy_metrics = {}
    coverage_metrics = {}
    for graspable, masked_graspables, partial_masks in zip(objects, masked_objects, all_progressive_masks):
        
        for mask in partial_masks:

            obj_name = graspable.key
            logging.info('Analyzing coverage for object %s' %(obj_name))

            # compute stable poses (for later down the pipeline, maybe we should remove this...)
            stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
            stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=config['min_prob'])
            stable_poses = stp.StablePoseFile().read(stable_pose_filename)

            # load the coverage data
            logging.info('Loading grasps')
            grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
            logging.info('Loading indices')
            indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
            logging.info('Loading quality')
            quality_filename = os.path.join(config['out_dir'], '{}_qualities.json'.format(obj_name))
            f = open(grasps_filename, 'r')
            grasps = pkl.load(f)
            for i, grasp in enumerate(grasps):
                grasp.grasp_id_ = i
            index_pairs = np.load(indices_filename)
            f = open(quality_filename, 'r')
            qualities = json.load(f)

            # get robust grasp qualties
            robust_qualities = []
            for quality in qualities:
                robust_qualities.append(quality['robust_force_closure'])

            # split covering set into those that contact the mask and those that don't
            public_grasps = []
            public_index_pairs = []
            public_qualities = []
            private_grasps = []
            private_index_pairs = []
            private_qualities = []
            for grasp, index_pair, quality in zip(grasps, index_pairs, robust_qualities):
                if index_pair[0] not in mask and index_pair[1] not in mask:
                    public_grasps.append(grasp)
                    public_index_pairs.append(index_pair)
                    public_qualities.append(quality)
                else:
                    private_grasps.append(grasp)
                    private_index_pairs.append(index_pair)
                    private_qualities.append(quality)                

            # compute metrics
            mn, mx = graspable.mesh.bounding_box()
            alpha = max(max(mx[2] - mn[2], mx[1] - mn[1]), mx[0] - mn[0])
            privacy_metric = graspable.mesh.surface_area(mask) / graspable.mesh.surface_area()
            coverage_metric = compute_coverage_metric(private_grasps, public_grasps, alpha=alpha)
            logging.info('Privacy metric: %f' %(privacy_metric))
            logging.info('Raw coverage metric: %f' %(coverage_metric))

            privacy_metrics[obj_name] = privacy_metric
            coverage_metrics[obj_name] = {}
            coverage_metrics[obj_name]['raw_coverage'] = coverage_metric

            # sort qualities
            public_grasps_and_qualities = zip(public_grasps, public_qualities)
            public_grasps_and_qualities.sort(key = lambda x: x[1], reverse = True)
            public_grasps = [gq[0] for gq in public_grasps_and_qualities]
            public_qualities = [gq[1] for gq in public_grasps_and_qualities]

            private_grasps_and_qualities = zip(private_grasps, private_qualities)
            private_grasps_and_qualities.sort(key = lambda x: x[1], reverse = True)
            private_grasps = [gq[0] for gq in private_grasps_and_qualities]
            private_qualities = [gq[1] for gq in private_grasps_and_qualities]

            # compute coverage metrics for increasing quality thresholds
            quality_res = config['quality_res']
            num_quality = 1.0  / quality_res
            quality_vals = quality_res * np.arange(1, num_quality)
            public_ind = len(public_grasps) - 1
            private_ind = len(private_grasps) - 1
            for tau in quality_vals:
                while public_ind >= 0 and public_qualities[public_ind] < tau:
                    public_ind = public_ind - 1
                while private_ind >= 0 and private_qualities[private_ind] < tau:
                    private_ind = private_ind - 1

                coverage_metric = compute_coverage_metric(private_grasps[:private_ind+1], public_grasps[:public_ind+1], alpha=alpha)
                coverage_metrics[obj_name]['tau=%.2f'%(tau)] = coverage_metric

            # visualize grasps with stable pose
            stable_pose_probs = np.array([s.p for s in stable_poses])
            highest_index = np.where(stable_pose_probs == np.max(stable_pose_probs))[0]
            most_stable_pose = stable_poses[highest_index[0]]

            # prune grasps in collision for the true object
            logging.info('Analyzing collision free grasps for original object')
            num_grasps = -1#100
            gripper = gr.RobotGripper.load(config['gripper'])
            private_coll_free_grasps, private_coll_free_qualities = \
                coverage.prune_grasps_in_collision(graspable, private_grasps[:num_grasps], private_qualities[:num_grasps],
                                                   gripper, most_stable_pose, config)
            public_coll_free_grasps, public_coll_free_qualities = \
                coverage.prune_grasps_in_collision(graspable, public_grasps[:num_grasps], public_qualities[:num_grasps],
                                                   gripper, most_stable_pose, config)
            coverage_metric = compute_coverage_metric(private_coll_free_grasps, public_coll_free_grasps, alpha=alpha)
            coverage_metrics[obj_name]['raw_coll_free'] = coverage_metric
            logging.info('Pruned to %s public collision free grasps' %(len(public_coll_free_grasps)))




    # read plotting params
    dpi = config['dpi']
    font_size = config['font_size']
    line_width = config['line_width']

    # store metrics
    privacy_met ric_filename = os.path.join(experiment_dir, 'privacy_metrics.json')
    f = open(privacy_metric_filename, 'w')
    json.dump(privacy_metrics, f)
    coverage_metric_filename = os.path.join(experiment_dir, 'coverage_metrics.json')
    f = open(coverage_metric_filename, 'w')
    json.dump(coverage_metrics, f)

    # plot the coverage metrics 
    plt.figure()
    plt.scatter(privacy_metrics.values(), coverage_metrics.values(), s=100, c='g')
    plt.xlabel('Privacy', fontsize=font_size)
    plt.ylabel('Coverage', fontsize=font_size)
    plt.show()
