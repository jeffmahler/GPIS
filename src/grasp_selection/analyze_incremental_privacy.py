"""
Script to visualize the computed coverage results
Author: Jeff Mahler
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython
import scipy.spatial as ss

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

def compute_volume_coverage_metric(grasp_universe, grasp_set, alpha=1.0):
    """ Coverage metric based on volume ratios """
    return float(len(grasp_universe)) / (len(grasp_universe) + len(grasp_set))

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
    all_original_masks = []
    all_tri_ind_mappings = []

    # setup experiment output dir
    experiment_id = gen_experiment_id()
    if config['load_experiment'] is not None:
        experiment_id = config['load_experiment']
    output_dir = config['out_dir']
    experiment_dir = os.path.join(output_dir, 'incremental-analysis-%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    logging.info('Saving analysis to %s' %(experiment_dir))

    # get the mode
    gripper = gr.RobotGripper.load(config['gripper'])
    mode = config['contact_point_mode']
    logging.info('Using contact point mode %d' %(mode))

    """
    num_views = config['num_views']
    delta_view = 360.0 / num_views
    elevs = [55.0, 30.0]
    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))

    # load all of the objects
    object_index = 0

    for object_key in object_keys:
        
        logging.info('Loading object %s' %(object_key))
        # hack -- fix
        #if object_key in ('gearbox', 'mount2', 'pipe_connector'):
        #    continue
        
        # first load the clean mesh
        mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+db.OBJ_EXT)).read()
        mesh, tri_ind_mapping = mesh.subdivide(min_triangle_length=config['min_tri_length'])

        sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+db.SDF_EXT)).read()
        current_object = go.GraspableObject3D(sdf, mesh, key=object_key)
        current_object.model_name_ = os.path.join(data_dir, object_key+db.OBJ_EXT)
        objects.append(current_object)

        #MASK LOADING
        original_object_masks = []
        progressive_object_masks = []

        for partial_mask_file in glob.glob(os.path.join(data_dir, object_key) + '_partial_mask*.npy'):
            current_partial_mask = np.load(partial_mask_file)
            
            sub_divided_mask = []
            for index in current_partial_mask.tolist():
                for new_index in tri_ind_mapping[index]:
                    sub_divided_mask.append(new_index)
            current_partial_mask = np.array(sub_divided_mask)

            progressive_mask = current_partial_mask
            for original_mask in original_object_masks:
                progressive_mask = np.append(original_mask, progressive_mask)

            original_object_masks.append(current_partial_mask)
            progressive_object_masks.append(progressive_mask)

        all_progressive_masks.append(progressive_object_masks)
        all_original_masks.append(original_object_masks)
        all_tri_ind_mappings.append(tri_ind_mapping)
        object_index += 1

    # loop through the objects and compute coverage for each
    privacy_metrics = {}
    coverage_metrics = {}
    for graspable, partial_masks, orig_partial_masks, tri_ind_mapping in \
            zip(objects, all_progressive_masks, all_original_masks, all_tri_ind_mappings):

        obj_name = graspable.key
        logging.info('Analyzing coverage for object %s' %(obj_name))

        # generate convex hulls for masked regions
        private_mesh_cvhs = []
        for i, mask in enumerate(orig_partial_masks):
            private_mesh = graspable.mesh.mask(mask)
            private_mesh_cvh = private_mesh.convex_hull()
            private_mesh_cvh.rescale(1.025, center=True) # rescaling for weird triangle flicker (which could be fixed with fancy zippering later)
            private_mesh_cvhs.append(private_mesh_cvh)

        # compute stable poses (for later down the pipeline, maybe we should remove this...)
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=config['min_prob'])
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)
        stable_poses.sort(key=lambda s: s.p, reverse=True)
        most_stable_pose = stable_poses[0]

        # load the coverage data
        logging.info('Loading grasps')
        grasps_filename = os.path.join(config['grasp_dir'], '{}_grasps.pkl'.format(obj_name))
        logging.info('Loading indices')
        indices_filename = os.path.join(config['grasp_dir'], '{}_indices.npy'.format(obj_name))
        logging.info('Loading quality')
        quality_filename = os.path.join(config['grasp_dir'], '{}_qualities.pkl'.format(obj_name))
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
        grasps_indices_qualities.sort(key=lambda x: x[2]['robust_force_closure'], reverse=True)
        #random.shuffle(grasps_indices_qualities)
        grasps_indices_qualities = grasps_indices_qualities[:num_grasps_to_analyze:5]
        grasps = [giq[0] for giq in grasps_indices_qualities]
        index_pairs = [giq[1] for giq in grasps_indices_qualities]
        qualities = [giq[2] for giq in grasps_indices_qualities]

        # get robust grasp qualties
        robust_qualities = []
        for quality in qualities:
            robust_qualities.append(quality['robust_force_closure'])

        # compute alpha for the coverage metric
        mn, mx = graspable.mesh.bounding_box()
        alpha = max(max(mx[2] - mn[2], mx[1] - mn[1]), mx[0] - mn[0])
        privacy_metrics[obj_name] = []
        coverage_metrics[obj_name] = {}
        coverage_metrics[obj_name]['volume'] = []
        coverage_metrics[obj_name]['distance'] = []

        # visualize meshes
        for i in range(len(private_mesh_cvhs)):
            # get mask
            mask = partial_masks[i]

            # compute public region
            public_mask = np.setdiff1d(np.arange(len(graspable.mesh.triangles())), mask)
            public_mesh = graspable.mesh.mask(np.array(public_mask))

            # compute public & private grasp sets
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

            # prune grasps in collision
            coll_free_grasps, coll_free_qualities = coverage.prune_grasps_in_collision(graspable, public_grasps, public_qualities,
                                                                                       gripper, most_stable_pose, config)


            # display coverage
            mlab.clf()
            coverage.vis_coverage(public_mesh, private_mesh_cvhs[:i+1], coll_free_grasps, coll_free_qualities, most_stable_pose,
                                  config, rank=False)
            for j, elev in enumerate(elevs):
                mlab.view(elevation=elev)
                for k in range(num_views):
                    az = k * delta_view
                    mlab.view(az)
                    figname = 'mesh_%s_partial%d_%d_%d.png' %(obj_name, i, j, k)
                    mlab.savefig(os.path.join(experiment_dir, figname))

            #mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
            #for j in range(i+1):
            #    mv.MayaviVisualizer.plot_stable_pose(private_mesh_cvhs[j], most_stable_pose, d=.15, color=(0,0,1))
            #mv.MayaviVisualizer.plot_stable_pose(public_mesh, most_stable_pose, d=.15, color=(0.6,0.6,0.6))
            #mlab.show()
        
        for i, mask in enumerate(partial_masks):
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

            # todo ---------------------------------------------------
            # plot public and colored private mesh
            progressive_mesh = graspable.mesh.mask(mask)
            public_mask = np.setdiff1d(np.arange(len(graspable.mesh.triangles())), mask)
            public_mesh = graspable.mesh.mask(public_mask)

            mlab.clf()
            # mv.MayaviVisualizer.plot_stable_pose(progressive_mesh, most_stable_pose, d=.15, color=(1,0,0))
            # mv.MayaviVisualizer.plot_stable_pose(public_mesh, most_stable_pose, d=.15)
            mv.MayaviVisualizer.plot_stable_pose(progressive_mesh, most_stable_pose, d=.15, color=(0,0,1))
            mv.MayaviVisualizer.plot_stable_pose(public_mesh, most_stable_pose, d=.15)
            for j, elev in enumerate(elevs):
                mlab.view(elevation=elev)
                for k in range(num_views):
                    az = k * delta_view
                    mlab.view(az)
                    # figname = 'mesh_%s_partial%d_%d_%d.png' %(obj_name, i, j, k)
                    mlab.savefig(os.path.join('figuring', 'table-outline_public-gray06_private-blue_%d_%d.png' %(j, k)))
                    # mlab.show()
                    # mlab.savefig(os.path.join(experiment_dir, figname))
            break

            # compute public collision free grasps and qualities
            public_coll_free_grasps, public_coll_free_qualities = \
                coverage.prune_grasps_in_collision(graspable, public_grasps, public_qualities,
                                                   gripper, most_stable_pose, config)

            # plot grasp wireframes
            mlab.clf()
            coverage.vis_coverage(public_mesh, progressive_mesh, public_coll_free_grasps, public_coll_free_qualities,
                                  most_stable_pose, config, gripper=None,
                                  max_display=config['max_display'], rank=False)
            for j, elev in enumerate(elevs):
                mlab.view(elevation=elev)
                for k in range(num_views):
                    az = k * delta_view
                    mlab.view(az)
                    figname = 'public_grasps_%s_partial%d_%d_%d.png' %(obj_name, i, j, k)
                    mlab.savefig(os.path.join(experiment_dir, figname))
            # end todo -------------------------------------------------

            if config['load_experiment'] is not None: # set in config to skip metric recomputation
                continue

            # compute metrics
            privacy_metric = graspable.mesh.surface_area(mask) / graspable.mesh.surface_area()
            logging.info('Privacy metric: %f' %(privacy_metric))
            volume_coverage_metric = compute_volume_coverage_metric(private_grasps, public_grasps, alpha=alpha)
            logging.info('Raw volume coverage metric: %f' %(volume_coverage_metric))
            coverage_metric = compute_coverage_metric(private_grasps, public_grasps, alpha=alpha)
            logging.info('Raw distance coverage metric: %f' %(coverage_metric))

            privacy_metrics[obj_name].append(privacy_metric)
            coverage_metrics[obj_name]['distance'].append(coverage_metric)
            coverage_metrics[obj_name]['volume'].append(volume_coverage_metric)

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
    """
    # read plotting params
    dpi = config['dpi']
    font_size = config['font_size']
    line_width = config['line_width']

    # store metrics
    privacy_metric_filename = os.path.join(experiment_dir, 'privacy_metrics.json')
    coverage_metric_filename = os.path.join(experiment_dir, 'coverage_metrics.json')
    if config['load_experiment'] is None:
        f = open(privacy_metric_filename, 'w')
        json.dump(privacy_metrics, f)
        f.close()
        f = open(coverage_metric_filename, 'w')
        json.dump(coverage_metrics, f)
        f.close()
    else:
        privacy_metrics = json.load(open(privacy_metric_filename, 'r'))
        coverage_metrics = json.load(open(coverage_metric_filename, 'r'))

    # plot the coverage metrics 
    plot_fmt = 'pdf'
    for obj_name in object_keys: 
        plt.figure()
        # plt.scatter(privacy_metrics[obj_name], coverage_metrics[obj_name]['distance'], s=100, c='g')
        plt.plot(privacy_metrics[obj_name], np.exp(-np.array(coverage_metrics[obj_name]['distance'])), c='g')
        plt.title('Incremental Privacy vs Exp Coverage for Object %s' %(obj_name))
        plt.xlabel('Privacy', fontsize=font_size)
        plt.ylabel('Coverage', fontsize=font_size)
        plt.xlim(0, 1)
        #plt.ylim(np.min(np.exp(-np.array(coverage_metrics[obj_name]['distance']))), np.max(np.exp(-np.array(coverage_metrics[obj_name]['distance']))))
        figname = 'incremental_privacy_vs_exp_cov_%s.%s' %(obj_name, plot_fmt)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi, format=plot_fmt)

        plt.figure()
        plt.plot(privacy_metrics[obj_name], coverage_metrics[obj_name]['distance'], c='g')
        plt.title('Incremental Privacy vs Exp Coverage for Object %s' %(obj_name))
        plt.xlabel('Privacy', fontsize=font_size)
        plt.ylabel('Coverage', fontsize=font_size)
        plt.xlim(0, 1)
        figname = 'incremental_privacy_vs_cov_%s.%s' %(obj_name, plot_fmt)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi, format=plot_fmt)

        plt.figure()
        # plt.scatter(privacy_metrics[obj_name], coverage_metrics[obj_name]['volume'], s=100, c='g')
        plt.plot(privacy_metrics[obj_name], coverage_metrics[obj_name]['volume'], c='g')
        plt.title('Incremental Privacy vs Volume Coverage for Object %s' %(obj_name))
        plt.xlabel('Privacy', fontsize=font_size)
        plt.ylabel('Volume Coverage', fontsize=font_size)
        plt.xlim(0, 1)
        figname = 'incremental_privacy_vs_vol_cov_%s.%s' %(obj_name, plot_fmt)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi, format=plot_fmt)
