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

def load_patches(obj, obj_id, dataset, all_features, all_metrics, all_obj_ids, config):
    grasps = dataset.grasps(obj.key)
    gm = dataset.grasp_metrics(obj.key, grasps)
    gf = dataset.grasp_features(obj.key, grasps)

    for grasp_id, features in gf.iteritems():

        # validate feature lengths
        metrics = gm[grasp_id]
        if len(features) == 0 or len(metrics.keys()) == 0:
            continue

        # parse metrics
        all_obj_ids.append(obj_id)

        for m, val in metrics.iteritems():
            if m not in all_metrics.keys():
                all_metrics[m]= []
            all_metrics[m].append(val)

        for feature in features:
            if feature.name not in all_features.keys():
                all_features[feature.name] = []
            all_features[feature.name].append(feature.descriptor)

    if len(all_features['w1_projection_disc']) != len(all_metrics[all_metrics.keys()[0]]):
        for key in all_features.keys():
            print key, len(all_features[key])
        for key in all_metrics.keys():
            print key, len(all_metrics[key])
        IPython.embed()
        exit(0)

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

    # make dataset output directory
    dest = args.output_dest
    try:
        os.makedirs(dest)
    except os.error:
        pass

    all_features = {}
    all_metrics = {}
    all_obj_ids = []

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # label each object in the dataset with grasps
        obj_id = 0
        for obj in dataset:
            logging.info('Loading patches for object {}'.format(obj.key))
            try:
                load_patches(obj, obj_id, dataset, all_features, all_metrics, all_obj_ids, config)
                obj_id = obj_id+1
            except Exception as e:
                logging.warning('Failed to complete grasp labelling for object {}'.format(obj.key))
                logging.warning(str(e))

    logging.info('Saving object ids')
    obj_id_array = np.array(all_obj_ids)
    filename = os.path.join(dest, 'obj_ids.npz')
    f = open(filename, 'w')
    np.savez_compressed(f, obj_id_array)

    for feature_name, feature_descriptors in all_features.iteritems():
        logging.info('Saving features %s' %(feature_name))

        descriptor_array = np.array(feature_descriptors)
        filename = os.path.join(dest, '%s.npz' %(feature_name))
        f = open(filename, 'w')
        np.savez_compressed(f, descriptor_array)

    for metric_name, metric_values in all_metrics.iteritems():
        logging.info('Saving metrics %s' %(metric_name))

        metric_array = np.array(metric_values)
        filename = os.path.join(dest, '%s.npz' %(metric_name))
        f = open(filename, 'w')
        np.savez_compressed(f, metric_array)

    database.close()
