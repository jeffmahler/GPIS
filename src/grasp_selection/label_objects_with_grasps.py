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
import termination_conditions as tc

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dest')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    IPython.embed()

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # make dataset output directory
        dest = os.path.join(args.output_dest, dataset.name)
        try:
            os.makedirs(dest)
        except os.error:
            pass
        output_db_filename = os.path.join(dest, config['results_database_name'])
        output_db = db.Hdf5Database(output_db_filename, config, access_level=db.WRITE_ACCESS)
        output_ds = output_db.create_dataset(dataset_name)

        # label each object in the dataset with grasps
        for obj in dataset:
            logging.info('Labelling object {} with grasps'.format(obj.key))
            output_ds.create_graspable(obj.key)
            label_grasps(obj, output_ds, config)

