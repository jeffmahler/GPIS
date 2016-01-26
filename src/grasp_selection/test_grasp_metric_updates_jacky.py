"""
Main file for labelling an object with "raw" grasps.
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
import numpy as np
#import scipy.stats

#import antipodal_grasp_sampler as ags
import database as db
import experiment_config as ec
'''
import discrete_adaptive_samplers as das
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
'''

if __name__ == '__main__':
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('dataset')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read the grasp metrics and features
    ds = database.dataset(args.dataset)
    o = ds.object_keys
    grasps = ds.grasps(o[0])
    grasp_features = ds.grasp_features(o[0], grasps)
    grasp_metrics = ds.grasp_metrics(o[0], grasps)

    IPython.embed()

    database.close()
