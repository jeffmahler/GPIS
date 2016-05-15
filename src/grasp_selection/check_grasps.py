"""
Script for visualizing / debugging the output of label_objects_with_grasps
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
import openravepy as rave
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import gripper as gr
import grasp_sampler as gs
import grasp_collision_checker as gcc
import json_serialization as jsons
import kernels
import mayavi_visualizer as mv
import models
import objectives
import obj_file
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def prune_grasps_in_collision(grasps, stable_pose):
    coll_free_grasps = []
    for i, grasp in enumerate(grasps):
        grasp1 = grasp.grasp_aligned_with_stable_pose(stable_pose)
        if not grasp1.collides_with_stable_pose(stable_pose):
            coll_free_grasps.append(grasp)
    return coll_free_grasps

def generate_candidate_grasps(grasps, grasp_metrics, num_candidates=10):
    random.shuffle(grasps)
    grasps = grasps[:num_candidates]
    return grasps

def show_stable_poses(obj, dataset, config, delay=0.1, num_views=16):
    # load grasps and stable poses
    stable_poses = dataset.stable_poses(obj.key)
    grasps = dataset.grasps(obj.key, gripper=config['gripper'])

    # display the stable poses
    T_table_world = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='world', to_frame='table')
    delta_view = 360.0 / num_views
    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
    for stable_pose in stable_poses:
        logging.info('Showing stable pose %s with p=%f' %(stable_pose.id, stable_pose.p))
        mlab.clf()
        mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, d=0.1, style='surface')

        #mlab.title('%s' %(stable_pose.id), color=(0,0,0))
        for j in range(num_views):
            az = j * delta_view
            mlab.view(az)
            time.sleep(delay)

def show_grasps(obj, dataset, config):
    # load grasps
    grasps = dataset.grasps(obj.key, gripper=config['gripper'])

    # plot mesh 
    T_table_world = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='world', to_frame='table')
    T_obj_table = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='table', to_frame='obj')
    T_obj_world = T_obj_table.dot(T_table_world)

    # plot grasps
    for grasp in grasps:
        logging.info('Displaying grasp %d of %d' %(grasp.grasp_id, len(grasps)))
        mlab.clf()
        mv.MayaviVisualizer.plot_mesh(obj.mesh, T_obj_world, style='surface')
        #mv.MayaviVisualizer.plot_grasp(grasp, T_obj_world, plot_approach=True, alpha=0.1, tube_radius=0.0025)
        mv.MayaviVisualizer.plot_gripper(grasp, T_obj_world)
        #time.sleep(1)
        mlab.show()

def show_grasps_on_stable_pose(obj, dataset, gripper, config, stable_pose_id='pose_0', delay=0.2,
                               num_grasp_views=8, save=False):
    # load grasps and stable poses
    stable_pose = dataset.stable_pose(obj.key, stable_pose_id)
    grasps = dataset.grasps(obj.key, gripper=gripper.name)

    # plot grasps on stable poses 
    T_table_world = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='world', to_frame='table')

    # collision checking
    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    collision_checker = gcc.OpenRaveGraspChecker(gripper, view=False)
    collision_checker.set_object(obj)

    # plot each grasp
    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))
    random.shuffle(grasps)
    delta_view = 360.0 / num_grasp_views
    for i, grasp in enumerate(grasps):
        # check collisions and plot if none
        grasp = grasp.parallel_table(stable_pose)

        if not gripper.collides_with_table(grasp, stable_pose) and not collision_checker.in_collision(grasp):
            # show metrics
            logging.info('Displaying grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps)))

            # plot
            mlab.clf()
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, d=0.1, style='surface',
                                                           color=(0.4,0.4,0.4))
            mv.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper=gripper, color=(0.65,0.65,0.65))
            mlab.show()

            for j in range(num_grasp_views):
                az = j * delta_view
                mlab.view(azimuth=az, focalpoint=(0,0,0), distance=0.15)
                time.sleep(delay)
                if save:
                    mlab.savefig(os.path.join(config['visualization_dir'], 'obj_%s_grasp_%d_gripper_%s_view_%d.png' %(obj.key, grasp.grasp_id, gripper.name, j)))

def show_grasps_by_metric(obj, dataset, config):
    # load grasps and stable poses
    stable_poses = dataset.stable_poses(obj.key)
    grasps = dataset.grasps(obj.key, gripper=config['gripper'])

    # plot grasps on stable poses 
    T_table_world = stf.SimilarityTransform3D(tfx.pose(np.eye(4)), from_frame='world', to_frame='table')
    T_obj_world = mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, style='surface')
    """
    """
    stable_pose = dataset.stable_pose(obj.key, 'pose_6')
    T_obj_world = mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, style='surface')
    for grasp in grasps:
        if grasp.grasp_id == 193:
            break
    mv.MayaviVisualizer.plot_grasp(grasp, T_obj_world, tube_radius=0.001)
    mlab.show()

    # load grasp metrics
    grasp_metrics = dataset.grasp_metrics(obj.key, grasps, gripper=config['gripper'])
    metric_names = grasp_metrics.values()[0].keys()

    metrics = ['force_closure', 'pfc_f_0.200000_tg_0.010000_rg_0.200000_to_0.010000_ro_0.200000']
    thresholds = [0, 0.1]
    high_quality_grasps = []
    for metric, tau in zip(metrics, thresholds):
        for grasp in grasps:
            if grasp_metrics[grasp.grasp_id][metric] > tau:
                high_quality_grasps.append(grasp)
    grasps = high_quality_grasps

    # get stable pose with highest probability
    probs = np.array([s.p for s in stable_poses])
    max_ind = np.where(probs == np.max(probs))[0][0]
    stable_pose = stable_poses[0]
    coll_free_grasps = prune_grasps_in_collision(grasps, stable_pose)

    # plot grasps
    grasps = generate_candidate_grasps(coll_free_grasps, grasp_metrics)
    grasp_ids = [g.grasp_id for g in grasps]
    for metric in metric_names:
        if metric.find(stable_pose.id) == -1 and metric.find('pfc') != -1 and metric.find('vfc') == -1 and metric.find('vpc') == -1: # ignore variance
            logging.info('Grasps colored by %s' %(metric))
            mlab.clf()

            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(obj.mesh, stable_pose, T_table_world, style='surface')

            metrics = np.array([grasp_metrics[i][metric] for i in grasp_ids])
            min_q = np.min(metrics)
            max_q = np.max(metrics)
            norm_metrics = 0.35 * (metrics - min_q) / (max_q - min_q)
            i = 0
            for grasp, q in zip(coll_free_grasps, norm_metrics.tolist()):
                logging.info('Displaying grasp %d (%d of %d)' %(grasp.grasp_id, i, len(grasps)))
                color = plt.get_cmap('hsv')(q)[:-1]
                g = grasp.grasp_aligned_with_stable_pose(stable_pose)
                mv.MayaviVisualizer.plot_grasp(g, T_obj_world, plot_approach=True, tube_radius=0.001,
                                               alpha=0.15, endpoint_color=color, grasp_axis_color=color, stp=stable_pose)
                i = i+1

            #mv.MayaviVisualizer.plot_colorbar(min_q, max_q)
            mlab.show()

if __name__ == '__main__':
    np.random.seed(100)
    random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_ONLY_ACCESS)
    gripper = gr.RobotGripper.load(config['grippers'][0])

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # check each object in the dataset with grasps
        for obj in dataset:
            logging.info('Displaying grasps for object {}'.format(obj.key))
            obj.model_name_ = dataset.obj_mesh_filename(obj.key)

            #show_stable_poses(obj, dataset, config)
            show_grasps_on_stable_pose(obj, dataset, gripper, config, config['ppc_stp_ids'][obj.key])

    database.close()

