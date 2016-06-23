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

def repair_metrics(old_metrics):
    metrics = {}
    for metric_name, metric_val in old_metrics.iteritems():
        metric_key = metric_name
        if metric_name.find('vpc') != -1:
            continue
        if metric_name.find('vfc') != -1:
            continue
        if metric_name.find('ppc') != -1:
            metric_key = metric_name[:4] + metric_name[11:]
        if metric_name.find('ewrr') != -1:
            metric_key = metric_name[:5] + metric_name[12:]
        if metric_name.find('lift_closure') != -1:
            metric_key = 'lift_closure'
        if metric_name.find('wrench_resist_ratio') != -1:
            metric_key = 'wrench_resist_ratio'
        metrics[metric_key] = metric_val
    return metrics

def load_patches(obj_key, obj_id, dataset, all_features, all_metrics,
                 all_obj_ids, all_names, config):
    normal_feature_name = 'surface_normals'
    grasp_id_feature_name = 'grasp_id'
    feature_names = [u'center',
                     u'moment_arms',
                     u'patch_orientation',
                     u'sampling_method',
                     u'surface_normals',
                     u'w1_curvature_window',
                     u'w1_gradx_window',
                     u'w1_grady_window',
                     u'w1_projection_window',
                     u'w2_curvature_window',
                     u'w2_gradx_window',
                     u'w2_grady_window',
                     u'w2_projection_window',
                     u'w1_curvature_disc',
                     u'w1_gradx_disc',
                     u'w1_grady_disc',
                     u'w1_projection_disc',
                     u'w2_curvature_disc',
                     u'w2_gradx_disc',
                     u'w2_grady_disc',
                     u'w2_projection_disc']

    logging.info('Loading grasps')
    grasps = dataset.grasps(obj_key, gripper=config['gripper'])
    logging.info('Loading grasp metrics')
    gm = dataset.grasp_metrics(obj_key, grasps, gripper=config['gripper'])
    logging.info('Loading grasp features')
    gf = dataset.grasp_features(obj_key, grasps, gripper=config['gripper'], feature_names=feature_names)
    logging.info('Num grasps: %d' %(len(grasps)))

    # create grasp id map for fixing normals
    grasp_map = {}
    for grasp in grasps:
        grasp_map[grasp.grasp_id] = grasp

    for grasp_id, features in gf.iteritems():
        # validity tag for removing NaNs
        data_valid = True

        # validate feature lengths
        metrics = repair_metrics(gm[grasp_id])
        if len(features) == 0 or len(metrics.keys()) == 0:
            continue

        # parse metrics
        all_obj_ids.append(obj_id)
        all_names.append('D-%s_O-%s' %(dataset.name, obj_key))

        for m, val in metrics.iteritems():
            if m not in all_metrics.keys():
                all_metrics[m]= []
            all_metrics[m].append(val)

        for feature in features:
            if feature.name not in all_features.keys():
                all_features[feature.name] = []
            all_features[feature.name].append(feature.descriptor)
            
            if np.isnan(np.sum(feature.descriptor)):
                data_valid = False

        # hack to get the surface normals
        if grasp_id_feature_name not in all_features.keys():
            all_features[grasp_id_feature_name] = []
        all_features[grasp_id_feature_name].append(grasp_id)

        # remove if data invalid
        if not data_valid:
            all_obj_ids.pop()
            all_names.pop()
            for feature_name in all_features.keys():
                all_features[feature_name].pop()
            for metric in all_metrics.keys():
                all_metrics[metric].pop()

    if len(all_features['w1_projection_window']) != len(all_metrics[all_metrics.keys()[0]]):
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

    num_datasets = len(config['datasets'].keys())
    all_features = {}
    all_metrics = {}
    all_obj_ids = []
    all_names = []
    data_id = 0
    obj_id = 0
    start_obj_id = 0
    end_obj_id = 10000000
    datapoints_per_file = 100
    threshold_obj_id = obj_id + datapoints_per_file

    for i, dataset_name in enumerate(config['datasets'].keys()):
        dataset = database.dataset(dataset_name)
        logging.info('Loading dataset %s' %(dataset.name))

        num_objects_remaining = len(dataset.object_keys)
        if i == num_datasets - 1 and (obj_id + num_objects_remaining) < threshold_obj_id:
            threshold_obj_id = obj_id + num_objects_remaining
        
        # label each object in the dataset with grasps
        for j, obj_key in enumerate(dataset.object_keys):
            logging.info('Loading patches for object {} with id {}'.format(obj_key, obj_id))
            if obj_id >= end_obj_id:
                database.close()
                exit(0)

            try:
                load_start = time.time()
                if obj_id >= start_obj_id:
                    load_patches(obj_key, obj_id, dataset, all_features, all_metrics,
                                 all_obj_ids, all_names, config)
                load_stop = time.time()
                logging.info('Patch loading took %f sec' %(load_stop - load_start))
                obj_id = obj_id+1
            except Exception as e:
                logging.warning('Failed to complete grasp labelling for object {}'.format(obj_key))
                logging.warning(str(e))
                
            if obj_id >= threshold_obj_id:
                if obj_id >= start_obj_id:
                    logging.info('Saving object ids')
                    obj_id_array = np.array(all_obj_ids)
                    filename = os.path.join(dest, 'obj_ids_%03d.npz' %(data_id))
                    f = open(filename, 'w')
                    np.savez_compressed(f, obj_id_array)

                    logging.info('Saving object names')
                    obj_name_array = np.array(all_names)
                    filename = os.path.join(dest, 'obj_names_%03d.npz' %(data_id))
                    f = open(filename, 'w')
                    np.savez_compressed(f, obj_name_array)

                    for feature_name, feature_descriptors in all_features.iteritems():
                        logging.info('Saving features %s' %(feature_name))

                        descriptor_array = np.array(feature_descriptors)
                        filename = os.path.join(dest, '%s_%03d.npz' %(feature_name, data_id))
                        f = open(filename, 'w')
                        np.savez_compressed(f, descriptor_array)
        
                    for metric_name, metric_values in all_metrics.iteritems():
                        logging.info('Saving metrics %s' %(metric_name))
                    
                        metric_array = np.array(metric_values)
                        filename = os.path.join(dest, '%s_%03d.npz' %(metric_name, data_id))
                        f = open(filename, 'w')
                        np.savez_compressed(f, metric_array)

                data_id = data_id+1
                del all_features
                del all_metrics
                del all_obj_ids
                del all_names
                all_features = {}
                all_metrics = {}
                all_obj_ids = []
                all_names = []

                threshold_obj_id = obj_id + datapoints_per_file
                num_objects_remaining = len(dataset.object_keys) - j
                if i == num_datasets - 1:
                    threshold_obj_id = obj_id + num_objects_remaining

    database.close()
