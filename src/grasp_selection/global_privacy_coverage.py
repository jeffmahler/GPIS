from __future__ import print_function

import logging
import os
import pickle as pkl
import sys

import numpy as np
import matplotlib.pyplot as plt

import experiment_config as ec
import grasp as g
import graspable_object as go
import obj_file as of
import sdf_file as sf
import quality as q

def load_graspable(path):
    name = path.split('/')[-1]
    print(path, name)
    sdf_path = path + '.sdf'
    obj_path = path + '.obj'
    graspable = go.GraspableObject3D(
        sf.SdfFile(sdf_path).read(), of.ObjFile(obj_path).read(), key=name)
    return graspable

def load_graspables(paths):
    graspables = []
    for path in paths:
        graspable = load_graspable(path)
        graspables.append(graspable)
    return graspables

def compute_actual_quality(obj, result_dir,
                           metric='ferrari_canny_L1', soft_fingers=True, friction_coef=0.5):
    actual_obj_path = os.path.join('masks', obj.key.replace('_dec', ''))
    grasp_path = os.path.join(result_dir, '{}_grasps.pkl'.format(obj.key))
    qualities_path = os.path.join(result_dir, '{}_qualities.npy'.format(obj.key))

    with open(grasp_path) as f:
        dec_grasps = pkl.load(f)
    dec_qualities = np.load(qualities_path)

    actual_obj = load_graspable(actual_obj_path)
    dec_on_actual_qualities = []
    for grasp in dec_grasps:
        dec_on_actual_quality = q.PointGraspMetrics3D.grasp_quality(
            grasp, actual_obj, metric, True, friction_coef)
        dec_on_actual_qualities.append(dec_on_actual_quality)
    dec_on_actual_qualities = np.array(dec_on_actual_qualities)
    
    dec_on_actual_qualities_path = qualities_path.replace('_qualities', '_actual_qualities')
    np.save(dec_on_actual_qualities_path, dec_on_actual_qualities)

    actual_qualities_path = qualities_path.replace('_dec', '')
    actual_qualities = np.load(actual_qualities_path)
    return dec_qualities, dec_on_actual_qualities, actual_qualities

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    out_dir = config['out_dir']
    paths = config['decimated_meshes']
    graspables = load_graspables(paths)
    qualities = []
    max_qual = 0
    for graspable in graspables:
        dec_qualities, dec_on_actual_qualities, actual_qualities = compute_actual_quality(graspable, out_dir)
        qualities.append((graspable.key, dec_qualities, dec_on_actual_qualities, actual_qualities))
        max_qual = max(max_qual, max(dec_qualities), max(dec_on_actual_qualities), max(actual_qualities))
    
    bins = np.linspace(0, max_qual, 10)
    for key, dec, dec_on_act, act in qualities:
        dec_weights = np.ones_like(dec) / float(len(dec))
        act_weights = np.ones_like(act) / float(len(act))
        plt.figure()
        plt.hist((dec, dec_on_act, act), bins,
                 weights=(dec_weights, dec_weights, act_weights),
                 # alpha=0.5,
                 label=('Estimated Quality', 'Actual Quality', 'Original Quality'))
        plt.ylim((0, 1))
        plt.legend(loc='upper right')
        plt.title('{} ({} grasps)'.format(key.replace('_dec', ''), len(dec)))
        plt.savefig('{}_quality_hist.png'.format(key), transparent=True)
        # plt.show()
