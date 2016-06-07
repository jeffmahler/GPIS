"""
Initial script for generating candidate grasps for execution on the physical robot
Author: Jeff Mahler
"""
import argparse
import copy
import logging
import pickle as pkl
import os
import openravepy as rave
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_collision_checker as gcc
import gripper as gr
import grasp_sampler as gs
import grasp as grasp_module
import json_serialization as jsons
import kernels
import mayavi_visualizer as mvis
import models
import objectives
import obj_file
import pfc
import plotting
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def collides_along_approach(grasp_candidate, gripper, collision_checker,
                            approach_dist, delta_approach):
    # check collisions along approach sequence
    grasp_pose = grasp_candidate.gripper_transform(gripper)
    collides = False
    cur_approach = 0
    grasp_approach_axis = grasp_pose.inverse().rotation[:,1]
    
    # check entire sequence
    while cur_approach <= approach_dist:
        grasp_approach_pose = copy.copy(grasp_pose.inverse())
        grasp_approach_pose.pose_ = tfx.pose(grasp_pose.inverse().rotation,
                                             grasp_pose.inverse().translation + cur_approach * grasp_approach_axis)
        
        if collision_checker.in_collision(grasp_approach_pose.inverse()):
            collides = True
            break
        cur_approach += delta_approach
        time.sleep(0.25)
    return collides

def generate_candidate_grasps(object_name, dataset, stable_pose,
                              num_grasps, gripper, config):
    """ Add the best grasp according to PFC as well as num_grasps-1 uniformly at random from the remaining set """
    grasp_set = []
    grasp_set_ids = []
    grasp_set_metrics = []

    # read params
    approach_dist = config['approach_dist']
    delta_approach = config['delta_approach']
    rotate_threshold = config['rotate_threshold']
    table_clearance = config['table_clearance']
    dist_thresh = config['grasp_dist_thresh']

    # get sorted list of grasps to ensure that we get the top grasp
    graspable = dataset.graspable(object_name)
    graspable.model_name_ = dataset.obj_mesh_filename(object_name)
    grasps = dataset.grasps(object_name, gripper=gripper.name)
    all_grasp_metrics = dataset.grasp_metrics(object_name, grasps, gripper=gripper.name)
    mn, mx = graspable.mesh.bounding_box()
    alpha = 1.0 / np.max(mx-mn)
    print alpha

    # prune by collisions
    rave.raveSetDebugLevel(rave.DebugLevel.Error)
    collision_checker = gcc.OpenRaveGraspChecker(gripper, view=False)
    collision_checker.set_object(graspable)

    # add the top quality grasps for each metric
    metrics = config['candidate_grasp_metrics']
    for metric in metrics:
        # generate metric tag
        if metric == 'efc':
            metric = db.generate_metric_tag('efcny_L1', config)
        elif metric == 'pfc':
            metric = db.generate_metric_tag('pfc', config)
        elif metric == 'ppc':
            metric = db.generate_metric_tag('ppc_%s' %(stable_pose.id), config)

        # sort grasps by the current metric
        grasp_metrics = [all_grasp_metrics[g.grasp_id][metric] for g in grasps]
        grasps_and_metrics = zip(grasps, grasp_metrics)
        grasps_and_metrics.sort(key = lambda x: x[1])
        grasps = [gm[0] for gm in grasps_and_metrics]
        grasp_metrics = [gm[1] for gm in grasps_and_metrics]

        # add grasps by quantile
        logging.info('Adding best grasp for metric %s' %(metric))
        i = len(grasps) - 1
        grasp_candidate = grasps[i].grasp_aligned_with_stable_pose(stable_pose)

        # check wrist rotation
        psi = grasp_candidate.angle_with_table(stable_pose)
        rotated_from_table = (psi > rotate_threshold)

        # check distances
        min_dist = np.inf
        for g in grasp_set:
            dist = grasp_module.ParallelJawPtGrasp3D.distance(g, grasp_candidate)
            if dist < min_dist:
                min_dist = dist

        # check collisions
        while gripper.collides_with_table(grasp_candidate, stable_pose, table_clearance) \
                or collides_along_approach(grasp_candidate, gripper, collision_checker, approach_dist, delta_approach) \
                or rotated_from_table or grasp_candidate.grasp_id in grasp_set_ids \
                or min_dist < dist_thresh:
            # get the next grasp
            i -= 1
            if i < 0:
                break
            grasp_candidate = grasps[i].grasp_aligned_with_stable_pose(stable_pose)

            # check wrist rotation
            psi = grasp_candidate.angle_with_table(stable_pose)
            rotated_from_table = (psi > rotate_threshold)

            # check distances
            min_dist = np.inf
            for g in grasp_set:
                dist = grasp_module.ParallelJawPtGrasp3D.distance(g, grasp_candidate)
                if dist < min_dist:
                    min_dist = dist

        # add to sequence
        if i >= 0:
            grasp_set.append(grasp_candidate)
            grasp_set_ids.append(grasp_candidate.grasp_id)
            grasp_set_metrics.append(all_grasp_metrics[grasp_candidate.grasp_id])

    # sample the remaining grasps uniformly at random
    i = 0
    random.shuffle(grasps)
    while len(grasp_set) < num_grasps and i < len(grasps):
        # random grasp candidate
        logging.info('Adding grasp %d' %(len(grasp_set)))
        grasp_candidate = grasps[i].grasp_aligned_with_stable_pose(stable_pose)

        # check wrist rotation
        psi = grasp_candidate.angle_with_table(stable_pose)
        rotated_from_table = (psi > rotate_threshold)

        # check distances
        min_dist = np.inf
        for g in grasp_set:
            dist = grasp_module.ParallelJawPtGrasp3D.distance(g, grasp_candidate)
            if dist < min_dist:
                min_dist = dist

        # check collisions
        while gripper.collides_with_table(grasp_candidate, stable_pose) \
                or collides_along_approach(grasp_candidate, gripper, collision_checker, approach_dist, delta_approach) \
                or rotated_from_table or grasp_candidate.grasp_id in grasp_set_ids \
                or min_dist < dist_thresh:
            # get the next grasp
            i += 1
            if i >= len(grasps):
                break
            grasp_candidate = grasps[i].grasp_aligned_with_stable_pose(stable_pose)

            # check wrist rotation
            psi = grasp_candidate.angle_with_table(stable_pose)
            rotated_from_table = (psi > rotate_threshold)

            # check distances
            min_dist = np.inf
            for g in grasp_set:
                dist = grasp_module.ParallelJawPtGrasp3D.distance(g, grasp_candidate)
                if dist < min_dist:
                    min_dist = dist

        # add to sequence
        if i < len(grasps):
            grasp_set.append(grasp_candidate)
            grasp_set_ids.append(grasp_candidate.grasp_id)
            grasp_set_metrics.append(all_grasp_metrics[grasp_candidate.grasp_id])

    return grasp_set, grasp_set_ids, grasp_set_metrics

if __name__ == '__main__':
    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('gripper')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_ONLY_ACCESS)

    gripper = gr.RobotGripper.load(args.gripper)

    # generate candidate grasps for each dataset
    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # make output dir
        candidate_output_dir = os.path.join(config['grasp_candidate_dir'],
                                            dataset_name)
        if not os.path.exists(candidate_output_dir):
            os.mkdir(candidate_output_dir)

        # generate grasps for each object in the dataset
        for obj in dataset:
            # read in object params
            object_key = obj.key
            if config['ppc_stp_ids']:
                stp_id = config['ppc_stp_ids'][object_key]
                stable_pose = dataset.stable_pose(object_key, stp_id)
            else:
                stable_poses = dataset.stable_poses(object_key)
                stable_pose = stable_poses[0]
                stp_id = stable_pose.id

            # check for existing candidates
            grasp_ids_filename = os.path.join(candidate_output_dir,
                '%s_%s_%s_grasp_ids.npy' %(object_key, stp_id, gripper.name))                                              
            if os.path.exists(grasp_ids_filename):
                logging.warning('Candidate grasps already exist for object %s in stable pose %s for gripper %s. Skipping...' %(object_key, stp_id, gripper.name))
                continue

            # generate candidate grasps
            logging.info('Computing candidate grasps for object {}'.format(object_key))
            candidate_grasp_set, candidate_grasp_ids, candidate_grasp_metrics = \
                generate_candidate_grasps(object_key, dataset, stable_pose,
                                          config['num_grasp_candidates'],
                                          gripper, config)

            # visualize grasp candidates
            T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
            T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')
            object_mesh = obj.mesh
            object_mesh_tf = object_mesh.transform(T_obj_stp)
            for i, grasp in enumerate(candidate_grasp_set):
                logging.info('')
                logging.info('Grasp %d (%d of %d)' %(grasp.grasp_id, i, len(candidate_grasp_set)))

                metrics = config['candidate_grasp_metrics']
                for metric in metrics:
                    if metric == 'efc':
                        metric = db.generate_metric_tag('efcny_L1', config)
                    elif metric == 'pfc':
                        metric = db.generate_metric_tag('pfc', config)
                    elif metric == 'ppc':
                        metric = db.generate_metric_tag('ppc_%s' %(stable_pose.id), config)
                    logging.info('Quality according to %s: %f' %(metric, candidate_grasp_metrics[i][metric]))

                mv.clf()
                T_obj_world = mvis.MayaviVisualizer.plot_stable_pose(object_mesh, stable_pose, T_table_world, d=0.1,
                                                                     style='surface', color=(1,0,0))
                mvis.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper=gripper)

                num_grasp_views = 8
                delta_view = 360.0 / num_grasp_views
                for j in range(num_grasp_views):
                    az = j * delta_view
                    mv.view(az)
                    time.sleep(0.5)

            # save candidate grasp ids
            np.save(grasp_ids_filename, candidate_grasp_ids)
            
    database.close()
