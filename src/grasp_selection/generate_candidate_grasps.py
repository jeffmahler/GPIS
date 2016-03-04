"""
Initial script for generating candidate grasps for execution on the physical robot
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
import mayavi.mlab as mlab
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import gripper as gr
import grasp_sampler as gs
import json_serialization as jsons
import kernels
import mayavi_visualizer as mv
import models
import objectives
import obj_file
import pfc
import plotting
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

from grasp_collision_checker import OpenRaveGraspChecker

def is_collision_free(obj, stable_pose, grasp, gripper):
    if gripper.collides_with_table(grasp, stable_pose):
        return False

    grasp_checker = OpenRaveGraspChecker(gripper_name = gripper.name)
    aligned_grasp = grasp.grasp_aligned_with_stable_pose(stable_pose)
            
    if grasp_checker.collides_with(obj, aligned_grasp):
        return False
        
    #TODO: hardcoded "flipping" aligned_grasp symmetrically to test collision for Zeke
    aligned_grasp.set_approach_angle(aligned_grasp.approach_angle + np.pi)
    if grasp_checker.collides_with(obj, aligned_grasp_mirrored):
        return False

    return True

def generate_candidate_grasps_quantile(obj, stable_pose, dataset, metric, num_grasps, config, grasp_set=[]):
    # load grasps and metrics
    sorted_grasps, sorted_metrics = dataset.sorted_grasps(obj.key, metric=metric, gripper=config['gripper'])

    grasp_ids = [g.grasp_id for g in grasp_set]

    # load gripper
    gripper = gr.RobotGripper.load(config['gripper'])

    # keep only the collision-free grasps
    cf_grasps = []
    cf_metrics = []
    for grasp, metric in zip(sorted_grasps, sorted_metrics):
        if is_collision_free(obj, stable_pose, grasp, gripper):
            cf_grasps.append(grasp)
            cf_metrics.append(metric)

    # get the quantiles
    num_cf_grasps = len(cf_grasps)
    delta_i = float(num_cf_grasps) / num_grasps
    delta_i = max(delta_i, 1)
    for i in range(num_grasps):
        j = int(i * delta_i)
        grasp = cf_grasps[j]
        while grasp.grasp_id in grasp_ids and j < num_cf_grasps:
            j = j+1
            grasp = cf_grasps[j]
        grasp_set.append(grasp)

    return grasp_set

def generate_candidate_grasps_random(obj, stable_pose, dataset, num_grasps, config, grasp_set=[]):
    # load grasps
    grasps = dataset.grasps(obj.key, gripper=config['gripper'])
    grasp_ids = [g.grasp_id for g in grasp_set]

    # load gripper
    gripper = gr.RobotGripper.load(config['gripper'])

    # keep only the collision-free grasps
    cf_grasps = []
    cf_metrics = []
    for grasp in grasps:
        if is_collision_free(obj, stable_pose, grasp, gripper):
            cf_grasps.append(grasp)

    # randomly sample grasps
    num_cf_grasps = len(cf_grasps)
    indices = np.arange(num_cf_grasps).tolist()
    random.shuffle(indices)
    i = 0
    j = 0
    while i < num_grasps and j < num_cf_grasps:
        grasp = cf_grasps[j]
        if grasp.grasp_id not in grasp_ids:
            grasp_set.append(grasp)
            i = i+1
        j = j+1
    return grasp_set

def generate_candidate_grasps_quantile_random(obj, stable_pose, dataset, config):
    metrics = config['metrics'].keys()
    num_grasps_per_metric = config['num_grasps_per_metric']
    num_random_grasps = config['num_random_grasps']
    
    min_q_vals = []
    max_q_vals = []
    for i in range(len(metrics)):
        min_q = config['metrics'][metrics[i]]['min_q']
        max_q = config['metrics'][metrics[i]]['max_q']
        min_q_vals.append(min_q)
        max_q_vals.append(max_q)
        if metrics[i].find('ppc') != -1:
            metrics[i] = metrics[i] %(stable_pose.id)
    
    # add quantiles for each desired metric
    grasp_set = []
    for metric in metrics:
        grasp_set = generate_candidate_grasps_quantile(obj, stable_pose, dataset, metric, num_grasps_per_metric, config, grasp_set=grasp_set)

    grasp_set = generate_candidate_grasps_random(obj, stable_pose, dataset, num_random_grasps, config, grasp_set=grasp_set)
        
    # plot a histogram for each metric
    grasp_metrics = dataset.grasp_metrics(obj.key, grasp_set, gripper=config['gripper'])
    for metric, min_q, max_q in zip(metrics, min_q_vals, max_q_vals):
        metric_vals = [grasp_metrics[g.grasp_id][metric] for g in grasp_set]
        plotting.plot_grasp_histogram(metric_vals, num_bins=config['num_bins'], min_q=min_q, max_q=max_q)
        figname = 'obj_%s_metric_%s_histogram.pdf' %(obj.key, metric)
        plt.savefig(os.path.join(config['output_dir'], figname), dpi=config['dpi'])

    grasp_ids = np.array([g.grasp_id for g in grasp_set])
    id_filename = 'obj_%s_%s_grasp_ids.npy' %(obj.key, stable_pose.id)
    np.save(os.path.join(config['output_dir'], id_filename), grasp_ids)

    # display grasps
    T_table_world = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='world', to_frame='table')
    gripper = gr.RobotGripper.load(config['gripper'])
    mlab.figure(bgcolor=(0.5,0.5,0.5), size=(1000,1000))
    delta_view = 360.0 / config['num_views']
    for i, grasp in enumerate(grasp_set):
        logging.info('Displaying grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasp_set)))

        for metric in metrics:
            logging.info('Metric %s = %.5f' %(metric.rjust(75), grasp_metrics[grasp.grasp_id][metric]))
        logging.info('')

        mlab.clf()
        T_obj_world = mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, d=0.1, style='surface')
        mv.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper=gripper, color=(1,1,1))

        for j in range(config['num_views']):
            az = j * delta_view
            mlab.view(az)
            time.sleep(config['view_delay'])

if __name__ == '__main__':
    np.random.seed(100)
    random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('object_key')
    parser.add_argument('stp_id')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_ONLY_ACCESS)
    object_key = args.object_key
    stp_id = args.stp_id

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # check each object in the dataset with grasps
        obj = dataset[object_key]
        stable_pose = dataset.stable_pose(object_key, stp_id)
        logging.info('Displaying grasps for object {}'.format(obj.key))
        generate_candidate_grasps_quantile_random(obj, stable_pose, dataset, config)
        
    database.close()
