B"""
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

def download_experiment_data(experiment_name, config):
    experiment_data_dir = 'gce_cache'
    if not os.path.exists(experiment_data_dir):
        os.mkdir(experiment_data_dir)

    download_data_command = 'gsutil -m cp gs://%s/%s*.tar.gz %s' %(config['compute']['bucket'], experiment_name, experiment_data_dir)
    os.system(download_data_command)

    tar_commands = []
    result_dirs = []
    for f in os.listdir(experiment_data_dir):
        local_file_root, ext = os.path.splitext(f)
        local_file_root, ext = os.path.splitext(local_file_root)
        filename = os.path.join(experiment_data_dir, f)
        result_dir = os.path.join(experiment_data_dir, local_file_root)

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        command = 'tar -xf %s -C %s' %(filename, result_dir)
        result_dirs.append(result_dir)
        tar_commands.append(command)
        
    for cmd in tar_commands:
        os.system(cmd)

    return result_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dest')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file and database
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_WRITE_ACCESS)
    logging.info('Database filename %s' %(database_filename))

    # download all experiment data
    result_dirs = download_experiment_data(config['job_root'], config)
    logging.info('Job root %s' %(config['job_root']))

    # for each experiment result, load and write to the database
    for result_dir in result_dirs:
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                if f.find(config['results_database_name']) != -1:
                    result_db_filename = os.path.join(root, config['results_database_name'])
                    logging.info('Result database filename %s' %(result_db_filename))
                    result_db = db.Hdf5Database(result_db_filename, config)

                    # write to dataset
                    result_datasets = result_db.datasets
                    for result_dataset in result_datasets:
                        dataset = database.dataset(result_dataset.name)
                        for obj_key in result_dataset.object_keys:
                            grasps = result_dataset.grasps(obj_key)
                            dataset.store_grasps(obj_key, grasps, force_overwrite=True)

                            grasp_feature_dict = result_dataset.grasp_features(obj_key, grasps)
                            dataset.store_grasp_features(obj_key, grasp_feature_dict, force_overwrite=True)

                            grasp_metric_dict = result_dataset.grasp_metrics(obj_key, grasps)
                            dataset.store_grasp_metrics(obj_key, grasp_metric_dict, force_overwrite=True) 
