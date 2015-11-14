"""
Main file for labelling an object with "raw" grasps.
Author: Jeff Mahler
"""
import argparse
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_sampler as gs
import json_serialization as jsons
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import random_variables as rvs
import robust_grasp_quality as rgq
import termination_conditions as tc

def plot_window_2d(window, num_steps, title='', save=False):
    """Plot a 2D image of a window."""
    if num_steps % 2 == 1: # center window at contact
        indices = np.array(range(-(num_steps // 2), (num_steps // 2) + 1))
    else: # arbitrarily cut off upper bound to preserve window size
        indices = np.array(range(-(num_steps // 2), (num_steps // 2)))
    indices = np.array(range(num_steps)) # for easier debugging

    fig = plt.figure()
    plt.title(title)
    imgplot = plt.imshow(window, extent=[indices[0], indices[-1], indices[-1], indices[0]],
                         interpolation='none', cmap=plt.cm.binary)
    plt.colorbar()
    plt.clim(-0.004, 0.004) # fixing color range for visual comparisons

    if save and title:
        plt.tight_layout()
        plt.savefig(title.replace(' ', '-'), bbox_inches='tight')
        plt.close()

def plot_disc(disc):
    ax = plt.gca(projection = '3d')    
    num_radial = disc.shape[0]
    num_angles = disc.shape[1]
    for i in range(num_angles):
        for r in range(num_radial):
            theta = 2.0 * np.pi * float(i) / num_angles
            x = r * np.cos(theta) 
            y = r * np.sin(theta) 
            ax.scatter(x, y, disc[r,i], s=100) 

def plot_window_and_disc(disc, proj_window):
    num_radial = disc.shape[0]
    num_angles = disc.shape[1]
    plt.figure()
    plt.imshow(proj_window, interpolation='none', cmap=plt.cm.binary)
    for i in range(num_angles):
        for r in range(num_radial):
            theta = 2.0 * np.pi * float(i) / num_angles
            x = r * np.cos(theta) + 6
            y = r * np.sin(theta) + 6
            plt.scatter(x, y, c=u'g') 

    plt.show()

def label_grasps(obj, output_ds, config):
    # sample grasps
    sample_start = time.clock()
    if config['grasp_sampler'] == 'antipodal':
        # antipodal sampling
        logging.info('Using antipodal grasp sampling')
        sampler = ags.AntipodalGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

        # pad with gaussian grasps
        num_grasps = len(grasps)
        min_num_grasps = config['min_num_grasps']
        if num_grasps < min_num_grasps:
            target_num_grasps = min_num_grasps - num_grasps
            gaussian_sampler = gs.GaussianGraspSampler(config)
            gaussian_grasps = gaussian_sampler.generate_grasps(
                obj, target_num_grasps=target_num_grasps, check_collisions=config['check_collisions'])
            grasps.extend(gaussian_grasps)
    else:
        # gaussian sampling
        logging.info('Using Gaussian grasp sampling')
        sampler = gs.GaussianGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

    sample_end = time.clock()
    sample_duration = sample_end - sample_start
    logging.info('Grasp candidate generation took %f sec' %(sample_duration))

    if not grasps or len(grasps) == 0:
        logging.info('Skipping %s' %(obj.key))
        return

    # store the grasps
    output_ds.store_grasps(obj.key, grasps)

    # re-load to get ids
    grasps = output_ds.grasps(obj.key)

    # extract features
    feature_extractor = ff.GraspableFeatureExtractor(obj, config)
    all_features = feature_extractor.compute_all_features(grasps)

    # store features
    output_ds.store_grasp_features(obj.key, grasps, all_features)
    f = output_ds.grasp_features(obj.key, grasps)

    # compute quality
    estimated_pfc = []
    estimated_q = []
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
    for i, grasp in enumerate(grasps):
        logging.info('Evaluating quality for grasp %d' %(i))
        grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, config)

        # probability of force closure
        pfc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric="force_closure")
        estimated_pfc.append(pfc)

        q = rgq.RobustGraspQuality.expected_quality(graspable_rv, grasp_rv, f_rv, config, quality_metric="ferrari_canny_L1")
        estimated_q.append(q)

    output_ds.store_grasp_metrics(obj.key, grasps, estimated_q, 'fc_L1')
    output_ds.store_grasp_metrics(obj.key, grasps, estimated_pfc, 'pfc')

if __name__ == '__main__':
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dest')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    output_db_filename = os.path.join(args.output_dest, config['results_database_name'])
    output_db = db.Hdf5Database(output_db_filename, config, access_level=db.WRITE_ACCESS)

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # make dataset output directory
        dest = os.path.join(args.output_dest, dataset.name)
        try:
            os.makedirs(dest)
        except os.error:
            pass
        output_ds = output_db.create_dataset(dataset_name)

        # label each object in the dataset with grasps
        for obj in dataset:
            logging.info('Labelling object {} with grasps'.format(obj.key))
            output_ds.create_graspable(obj.key)
            label_grasps(obj, output_ds, config)

    database.close()
    output_db.close()
