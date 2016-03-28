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

import csv
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

if __name__ == '__main__':
    data_dir = sys.argv[1]

    # store metrics
    privacy_metric_filename = os.path.join(data_dir, 'privacy_metrics.json')
    f = open(privacy_metric_filename, 'r')
    privacy_metrics = json.load(f)
    f.close()
    coverage_metric_filename = os.path.join(data_dir, 'coverage_metrics.json')
    f = open(coverage_metric_filename, 'r')
    coverage_metrics = json.load(f)
    f.close()

    # csv of the pct invalid
    pct_invalid_dicts = []
    for object_key in coverage_metrics.keys():
        pct_invalid_dict = {}
        pct_invalid_dict['object_key'] = object_key
        for masked_tag in masked_object_tags:
            tag = 'pct_invalid_grasps%s'%(masked_tag)
            pct_invalid_dict[tag] = coverage_metrics[object_key][tag]
        pct_invalid_dicts.append(pct_invalid_dict)

    pct_invalid_filename = os.path.join(data_dir, 'pct_invalid_grasps.csv')
    f = open(pct_invalid_filename, 'w')
    csv_writer = csv.DictWriter(f, pct_invalid_dicts[0].keys())
    csv_writer.writeheader()
    for pct_invalid_dict in pct_invalid_dicts:
        csv_writer.writerow(pct_invalid_dict)
    f.close()

    # scatter all collision-free coverage vs privacy
    tag = 'raw_coll_free'
    coverage_vals = {}
    coverage_vals[tag] = []
    for masked_tag in masked_object_tags:
        coverage_vals['raw_coll_free%s'%(masked_tag)] = []
    for object_key in coverage_metrics.keys():
        coverage_vals[tag].append(coverage_metrics[object_key][tag])
        for masked_tag in masked_object_tags:
            coverage_vals['raw_coll_free%s'%(masked_tag)].append(coverage_metrics[object_key]['raw_coll_free%s'%(masked_tag)])

    font_size = 15
    dpi = 400

    colors = ['r', 'g', 'b', 'c']
    labels = []

    plt.figure()
    for tag, coverage, color in zip(coverage_vals.keys(), coverage_vals.values(), colors):
        plt.scatter(coverage, privacy_metrics.values(), c=color, s=150)
        labels.append(tag)
    plt.xlabel('Coverage', fontsize=font_size)
    plt.ylabel('Privacy', fontsize=font_size)
    plt.legend(labels, loc='best')
    figname = 'coverage_vs_privacy.png'
    plt.savefig(os.path.join(data_dir, figname), dpi=dpi)

    plt.figure()
    for tag, coverage, color in zip(coverage_vals.keys(), coverage_vals.values(), colors):
        plt.scatter(coverage, privacy_metrics.values(), c=color, s=150)
        labels.append(tag)
    plt.xlabel('Coverage', fontsize=font_size)
    plt.ylabel('Privacy', fontsize=font_size)
    plt.legend(labels, loc='best')
    figname = 'privacy_vs_coverage.png'
    plt.savefig(os.path.join(data_dir, figname), dpi=dpi)

    priv_avg_cov = {}
    [priv_avg_cov.update({k: np.mean(v)}) for k, v, in coverage_vals.iteritems()]
    priv_avg_cov_filename = os.path.join(data_dir, 'priv_avg_cov.csv')
    f = open(priv_avg_cov_filename, 'w')
    csv_writer = csv.DictWriter(f, priv_avg_cov.keys())
    csv_writer.writeheader()
    csv_writer.writerow(priv_avg_cov)
    f.close()

    # plot scaling of coverage versus tau?

    IPython.embed()
    
