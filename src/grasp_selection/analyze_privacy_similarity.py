"""
Script to analyze the effects of masking techniques on similarity metrics
Author: Jeff Mahler
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython

from mayavi import mlab

import csv
import os
import json
import logging
import pickle as pkl
import random
import shutil
import scipy.spatial.distance as ssd
import sys
import time
sys.path.insert(0, 'src/grasp_selection/feature_vectors/')

import similarity_tf as stf
import tfx

import contacts
import coverage
import privacy_coverage as coverage
import database as db
import experiment_config as ec
from feature_database import FeatureDatabase
import gripper as gr
import grasp as g
import graspable_object as go
import mvcnn_feature_extractor as mvcnn_fex
import mayavi_visualizer as mv
import obj_file
import quality as q
import sdf_file
import similarity_tf as stf
import stp_file as stp
import random

masked_object_tags = ['no_mask', 'masked_bbox', 'masked_hull']

class FakeGraspable:
    def __init__(self, key):
        self.key = key

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    # load the models
    data_dir = config['data_dir']
    out_dir = config['out_dir']
    object_keys = coverage.read_index(data_dir)

    new_object_keys = []
    for object_key in object_keys:
        new_object_keys.append(object_key)
        for masked_tag in masked_object_tags:
            new_object_keys.append(object_key+'_'+masked_tag)

    graspables = [FakeGraspable(k) for k in new_object_keys]
    feature_db = FeatureDatabase(config)
    nearest_features = feature_db.nearest_features(name=config['priors_feature_names'][0])
    feature_extractor = mvcnn_fex.MVCNNBatchFeatureExtractor(config)
    feature_vectors = feature_extractor.extract(graspables)

    proj_feature_vectors = []
    for feature_vector in feature_vectors:
        proj_feature_vectors.append(nearest_features.project_feature_vector(feature_vector.descriptor))
        
    proj_feature_vectors = np.array(proj_feature_vectors)
    feature_dists = ssd.pdist(proj_feature_vectors)
    feature_dists = ssd.squareform(feature_dists)

    start_ind = 0
    end_ind = start_ind + len(masked_object_tags) + 1
    masked_object_dists = np.zeros([len(object_keys), len(masked_object_tags)])
    for i, object_key in enumerate(object_keys):
        d = feature_dists[start_ind:end_ind,start_ind:end_ind]
        masked_object_dists[i,:] = d[1:len(masked_object_tags)+1,0]
        start_ind = end_ind
        end_ind = start_ind + len(masked_object_tags) + 1 

    headers = ['object_key'] 
    for mask_tag in masked_object_tags:
        headers.append('distance_%s' %(mask_tag))

    priv_sim_filename = os.path.join(out_dir, 'priv_sim.csv')
    f = open(priv_sim_filename, 'w')
    csv_writer = csv.DictWriter(f, headers)
    csv_writer.writeheader()
    for i, object_key in enumerate(object_keys):
        row = {'object_key':object_key}
        for j, mask_tag in enumerate(masked_object_tags):
            row['distance_%s' %(mask_tag)] = masked_object_dists[i,j]
        csv_writer.writerow(row)
    f.close()

    IPython.embed()
