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

masked_object_tags = ['_no_mask', '_masked_bbox', '_masked_hull']

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
    experiment_dir = os.path.join(output_dir, 'analysis-%s' %(experiment_id))
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
        most_stable_pose = stable_poses[highest_index[0]]
        if graspable.key == 'mount2':
            most_stable_pose = stable_poses[2]
        if graspable.key == 'endstop_holder2':
            most_stable_pose = stable_poses[-1]

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
            coverage_metrics[obj_name]['raw_tau=%.2f'%(tau)] = coverage_metric

        # prune grasps in collision for the true object
        logging.info('Analyzing collision free grasps for original object')
        private_coll_free_grasps, private_coll_free_qualities = \
            coverage.prune_grasps_in_collision(graspable, private_grasps, private_qualities,
                                               gripper, most_stable_pose, config)
        public_coll_free_grasps, public_coll_free_qualities = \
            coverage.prune_grasps_in_collision(graspable, public_grasps, public_qualities,
                                               gripper, most_stable_pose, config)
        coverage_metric = compute_coverage_metric(private_coll_free_grasps, public_coll_free_grasps, alpha=alpha)
        coverage_metrics[obj_name]['raw_coll_free'] = coverage_metric
        logging.info('Pruned to %s public collision free grasps' %(len(public_coll_free_grasps)))

        # compute coverage metrics for increasing quality thresholds over collision free grasps
        public_ind = len(public_coll_free_grasps) - 1
        private_ind = len(private_coll_free_grasps) - 1
        for tau in quality_vals:
            while public_ind >= 0 and public_coll_free_qualities[public_ind] < tau:
                public_ind = public_ind - 1
            while private_ind >= 0 and private_coll_free_qualities[private_ind] < tau:
                private_ind = private_ind - 1

            coverage_metric = compute_coverage_metric(private_coll_free_grasps[:private_ind+1], public_coll_free_grasps[:public_ind+1], alpha=alpha)
            coverage_metrics[obj_name]['coll_free_tau=%.2f'%(tau)] = coverage_metric

        # plotting
        public_mask = []
        for i in range(len(graspable.mesh.triangles())):
            if i not in mask.tolist():
                public_mask.append(i)
        private_mesh = graspable.mesh.mask(mask)
        private_mesh_cvh = private_mesh.convex_hull()
        private_mesh_cvh.rescale(1.025, center=True) # rescaling for weird triangle flicker (which could be fixed with fancy zippering later)
        public_mesh = graspable.mesh.mask(np.array(public_mask))
        logging.info('Displaying grasps')

        # plot raw mesh
        mlab.clf()
        T_obj_world = mv.MayaviVisualizer.plot_stable_pose(graspable.mesh, most_stable_pose, d=0.15)
        """
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'mesh_%s_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))
        """

        # wireframe diagrams
        mlab.clf()
        coverage.vis_coverage(public_mesh, [private_mesh], public_coll_free_grasps, public_coll_free_qualities,
                              most_stable_pose, config, gripper=None, color_grasps=False,
                              max_display=config['max_display'], rank=False)
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'uncolored_public_grasps_%s_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))

        mlab.clf()
        coverage.vis_coverage(public_mesh, [private_mesh], public_coll_free_grasps, public_coll_free_qualities,
                              most_stable_pose, config, gripper=None,
                              max_display=config['max_display'], rank=False)
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'public_grasps_%s_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))

        mlab.clf()
        coverage.vis_coverage(public_mesh, [private_mesh_cvh], public_coll_free_grasps, public_coll_free_qualities,
                              most_stable_pose, config, gripper=None,
                              max_display=config['max_display'], rank=True)
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'public_grasps_%s_sorted_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))


        orig_coll_free_grasps = private_coll_free_grasps + public_coll_free_grasps
        orig_coll_free_qualities = private_coll_free_qualities + public_coll_free_qualities
        mlab.clf()
        coverage.vis_coverage(graspable.mesh, [], orig_coll_free_grasps, orig_coll_free_qualities,
                              most_stable_pose, config, gripper=None,
                              max_display=config['max_display'], rank=True)
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'all_coll_free_grasps_%s_sorted_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))

        mlab.clf()
        coverage.vis_coverage(graspable.mesh, [], orig_coll_free_grasps, orig_coll_free_qualities,
                              most_stable_pose, config, gripper=None,
                              max_display=config['max_display'], rank=False)
        for i, elev in enumerate(elevs):
            mlab.view(elevation=elev)
            for j in range(num_views):
                az = j * delta_view
                mlab.view(az)
                figname = 'all_coll_free_grasps_%s_view_%d_%d.png' %(obj_name, i, j)                
                mlab.savefig(os.path.join(experiment_dir, figname))

        # HACK: don't need to do further analysis for this run
        continue

        # plot individual grasps with grippers
        grasps_and_qualities = zip(public_coll_free_grasps, public_coll_free_qualities)
        grasps_and_qualities.sort(key=lambda x: x[1], reverse=True)
        sorted_public_coll_free_grasps = [gq[0] for gq in grasps_and_qualities]
        sorted_public_coll_free_qualities = [gq[1] for gq in grasps_and_qualities]
        step = max(len(sorted_public_coll_free_grasps) / config['max_gripper_display'], 1)
        for k in range(0, len(sorted_public_coll_free_grasps), step):
            #mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
            mlab.clf()
            coverage.vis_coverage(public_mesh, [private_mesh_cvh], [sorted_public_coll_free_grasps[k]], [sorted_public_coll_free_qualities[k]],
                                  most_stable_pose, config, gripper=gripper,
                                  max_display=config['max_display'], rank=False)
            #mlab.show()
            for j in range(num_views):
                az = j * delta_view
                mlab.view(azimuth=az, elevation=elevs[0], distance=config['cam_distance'], focalpoint=T_obj_world.inverse().translation)
                figname = 'public_grasp_%d_%s_robustness_%.3f_gripper_view_%d.png' %(k, obj_name, sorted_public_coll_free_qualities[k], j)
                mlab.savefig(os.path.join(experiment_dir, figname))

        # concatenate all collision free grasps and qualities for collision pruning on the other models
        orig_coll_free_grasps = private_coll_free_grasps + public_coll_free_grasps
        orig_coll_free_qualities = private_coll_free_qualities + public_coll_free_qualities
        orig_coll_free_grasp_ids = [grasp.grasp_id for grasp in orig_coll_free_grasps]

        for masked_graspable, mask_tag in zip(masked_graspables, masked_object_tags):
            # prune grasps in collision for masked objects
            logging.info('Analyzing collision free grasps for %s' %(masked_graspable.key))

            # NOTE: no need to re-prune private grasps bc we want to compare with the original object
            public_coll_free_grasps, public_coll_free_qualities = \
                coverage.prune_grasps_in_collision(masked_graspable, public_grasps, public_qualities,
                                                   gripper, most_stable_pose, config)
            coverage_metric = compute_coverage_metric(orig_coll_free_grasps, public_coll_free_grasps, alpha=alpha)
            coverage_metrics[obj_name]['raw_coll_free%s' %(mask_tag)] = coverage_metric
            logging.info('Pruned to %s public collision free grasps' %(len(public_coll_free_grasps)))

            # get percentage of grasps that are in collision on the original object
            public_coll_free_grasp_ids = [grasp.grasp_id for grasp in public_coll_free_grasps]
            num_invalid_grasps = 0
            for grasp_id in public_coll_free_grasp_ids:
                if grasp_id not in orig_coll_free_grasp_ids:
                    num_invalid_grasps = num_invalid_grasps+1
            pct_invalid_grasps = float(num_invalid_grasps) / float(len(public_coll_free_grasp_ids))
            coverage_metrics[obj_name]['pct_invalid_grasps%s' %(mask_tag)] = pct_invalid_grasps

            # compute coverage metrics for increasing quality thresholds over collision free grasps
            public_ind = len(public_coll_free_grasps) - 1
            private_ind = len(orig_coll_free_grasps) - 1
            for tau in quality_vals:
                while public_ind >= 0 and public_coll_free_qualities[public_ind] < tau:
                    public_ind = public_ind - 1
                while private_ind >= 0 and orig_coll_free_qualities[private_ind] < tau:
                    private_ind = private_ind - 1

                coverage_metric = compute_coverage_metric(orig_coll_free_grasps[:private_ind+1], public_coll_free_grasps[:public_ind+1], alpha=alpha)
                coverage_metrics[obj_name]['coll_free%s_tau=%.2f'%(mask_tag, tau)] = coverage_metric     

                coverage_metric = compute_coverage_metric(orig_coll_free_grasps, public_coll_free_grasps[:public_ind+1], alpha=alpha)
                coverage_metrics[obj_name]['coll_free%s_robust_tau=%.2f'%(mask_tag, tau)] = coverage_metric     

    # store metrics
    privacy_metric_filename = os.path.join(experiment_dir, 'privacy_metrics.json')
    f = open(privacy_metric_filename, 'w')
    json.dump(privacy_metrics, f)
    f.close()
    coverage_metric_filename = os.path.join(experiment_dir, 'coverage_metrics.json')
    f = open(coverage_metric_filename, 'w')
    json.dump(coverage_metrics, f)
    f.close()

    """
    for obj_key in object_keys:
        taus = []
        coverages = []
        for metric, val in coverage_metrics[obj_key].iteritems():
            if metric.find('hull') != -1 and metric.find('tau') != -1:
                taus.append(float(metric[-2:]))
                coverages.append(np.exp(-val))

        plt.figure()
        plt.plot(taus, coverages, s=100, c='g')
        plt.xlabel('Robustness', fontsize=font_size)
        plt.ylabel('Coverage', fontsize=font_size)
        plt.title('Privacy vs Coverage %s' %(metric_name), fontsize=font_size)
        figname = 'privacy_vs_cov_%s.png' %(metric_name)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi)
    """

    """
    # plot the coverage metrics 
    for metric_name in coverage_metrics[obj_name].keys():
        raw_coverage_metrics = []
        raw_privacy_metrics = []
        for obj_name, metrics in coverage_metrics.iteritems():
            if not np.isinf(metrics[metric_name]):
                raw_coverage_metrics.append(metrics[metric_name])
                raw_privacy_metrics.append(privacy_metrics[obj_name])
                
        plt.figure()
        plt.scatter(raw_privacy_metrics, raw_coverage_metrics, s=100, c='g')
        plt.xlabel('Privacy', fontsize=font_size)
        plt.ylabel('Coverage', fontsize=font_size)
        plt.title('Privacy vs Coverage %s' %(metric_name), fontsize=font_size)
        figname = 'privacy_vs_cov_%s.png' %(metric_name)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi)
    """
