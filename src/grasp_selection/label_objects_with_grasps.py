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
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_collision_checker as gcc
import grasp_sampler as gs
import gripper as gr
import json_serialization as jsons
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

GRAVITY_ACCEL = 9.81

def label_grasps(obj, dataset, output_ds, gripper_name, config):
    start_time = time.time()
    
    # sample grasps
    if config['sample_grasps']:
        logging.info('Generating new grasps')
        sample_start_time = time.time()

        if config['check_collisions']:
            obj.model_name_ = dataset.obj_mesh_filename(obj.key)

        sample_start = time.clock()
        if config['grasp_sampler'] == 'antipodal':
            # antipodal sampling
            logging.info('Using antipodal grasp sampling')
            sampler = ags.AntipodalGraspSampler(config)
            grasps = sampler.generate_grasps(
                obj, check_collisions=config['check_collisions'],
                max_iter=config['max_grasp_sampling_iters'])
            grasp_sms = [0] * len(grasps)

            # pad with gaussian grasps
            num_grasps = len(grasps)
            min_num_grasps = config['target_num_grasps']
            if min_num_grasps is None:
                min_num_grasps = config['min_num_grasps']

            if num_grasps < min_num_grasps:
                target_num_grasps = min_num_grasps - num_grasps
                gaussian_sampler = gs.GaussianGraspSampler(config)
                gaussian_grasps = gaussian_sampler.generate_grasps(
                    obj, target_num_grasps=target_num_grasps,
                    check_collisions=config['check_collisions'],
                    max_iter=config['max_grasp_sampling_iters'])

                grasps.extend(gaussian_grasps)
                grasp_sms.extend([1] * len(gaussian_grasps))
        elif config['grasp_sampler'] == 'anti_uniform':
            # half uniform and half antipodal sampling
            num_grasps_sample = config['target_num_grasps'] / 2

            antipodal_sampler = ags.AntipodalGraspSampler(config)
            antipodal_grasps = antipodal_sampler.generate_grasps(
                obj, target_num_grasps=num_grasps_sample,
                check_collisions=config['check_collisions'],
                max_iter=config['max_grasp_sampling_iters'])
            grasp_sms = [0] * len(antipodal_grasps)

            uniform_sampler = gs.UniformGraspSampler(config)
            uniform_grasps = uniform_sampler.generate_grasps(
                obj, target_num_grasps=num_grasps_sample,
                check_collisions=config['check_collisions'],
                max_iter=config['max_grasp_sampling_iters'])

            grasps = antipodal_grasps + uniform_grasps
            logging.info('Num antipodal %d' %(len(antipodal_grasps)))
            logging.info('Num uniform %d' %(len(uniform_grasps)))
            grasp_sms.extend([2] * len(uniform_grasps))
        else:
            # gaussian sampling
            logging.info('Using Gaussian grasp sampling')
            sampler = gs.GaussianGraspSampler(config)
            grasps = sampler.generate_grasps(
                obj, check_collisions=config['check_collisions'])
            grasp_sms.extend([1] * len(grasps))

        sample_end = time.clock()
        sample_duration = sample_end - sample_start
        logging.info('Grasp candidate generation took %f sec' %(sample_duration))

        if not grasps or len(grasps) == 0:
            logging.info('No grasps found for %s' %(obj.key))
            return

        # store the grasps
        output_ds.store_grasps(obj.key, grasps, gripper=gripper_name)

        sample_stop_time = time.time()
        logging.info('Sampling %d grasps took %f sec.' %(len(grasps), sample_stop_time - sample_start_time))

    # load all grasps to get ids
    grasps = output_ds.grasps(obj.key, gripper=gripper_name)

    # extract features
    feature_start_time = time.time()
    all_features = [None] * len(grasps)
    if config['extract_features']:
        logging.info('Extracting features')
        feature_extractor = ff.GraspableFeatureExtractor(obj, config)
        all_features = feature_extractor.compute_all_features(grasps)

    raw_feature_dict = {}
    for g, f, s in zip(grasps, all_features, grasp_sms):
        raw_feature_dict[g.grasp_id] = [ff.GraspFeature('sampling_method', 'sampling_method', s)]
        if f is not None:
            raw_feature_dict[g.grasp_id].append(f.features())
                
    # store features
    output_ds.store_grasp_features(obj.key, raw_feature_dict, gripper=gripper_name, force_overwrite=True)

    feature_stop_time = time.time()
    logging.info('Feature extraction for %d grasps took %f sec.' %(len(grasps), feature_stop_time - feature_start_time))

    # compute grasp metrics
    if config['compute_grasp_metrics']:
        logging.info('Computing grasp metrics')
        quality_start_time = time.time()

        # stable poses
        if config['ppc_stp_ids']:
            stable_poses = [dataset.stable_pose(obj.key, config['ppc_stp_ids'][obj.key])]
        else:
            stable_poses = dataset.stable_poses(obj.key, min_p=config['stp_min_p'])

        #convert stable poses to wrenches
        stable_pose_wrenches = []
        mass = config['object_mass'] #TODO: possibly change to real mass in the future
        gravity_magnitude = mass * GRAVITY_ACCEL
        m = obj.mesh
        for stable_pose in stable_poses:
            stable_pose_normal = stable_pose.r[2]
            gravity_force = -gravity_magnitude * stable_pose_normal
            wrench = np.append(gravity_force, [0,0,0])
            stable_pose_wrenches.append(-wrench)
        
        grasp_force_limit = config['grasp_force_limit']
        
        # create alternate configs for double and half the uncertainty
        low_u_mult = config['low_u_mult']
        high_u_mult = config['high_u_mult']
        low_u_config = copy.deepcopy(config)

        low_u_config['sigma_mu'] = low_u_mult * low_u_config['sigma_mu'] 
        low_u_config['sigma_rot_grasp'] = low_u_mult * low_u_config['sigma_rot_grasp'] 
        low_u_config['sigma_trans_grasp'] = low_u_mult * low_u_config['sigma_trans_grasp'] 
        low_u_config['sigma_rot_obj'] = low_u_mult * low_u_config['sigma_rot_obj'] 
        low_u_config['sigma_trans_obj'] = low_u_mult * low_u_config['sigma_trans_obj'] 
        low_u_config['sigma_scale_obj'] = low_u_mult * low_u_config['sigma_scale_obj'] 

        med_u_config = copy.deepcopy(config)

        high_u_config = copy.deepcopy(config)
        high_u_config['sigma_mu'] = high_u_mult * high_u_config['sigma_mu'] 
        high_u_config['sigma_rot_grasp'] = high_u_mult * high_u_config['sigma_rot_grasp'] 
        high_u_config['sigma_trans_grasp'] = high_u_mult * high_u_config['sigma_trans_grasp'] 
        high_u_config['sigma_rot_obj'] = high_u_mult * high_u_config['sigma_rot_obj'] 
        high_u_config['sigma_trans_obj'] = high_u_mult * high_u_config['sigma_trans_obj'] 
        high_u_config['sigma_scale_obj'] = high_u_mult * high_u_config['sigma_scale_obj'] 

        # compute deterministic quality
        grasp_metrics = {}
        logging.info('Computing deterministic metrics')
        for i, grasp in enumerate(grasps):
            logging.info('Evaluating deterministic quality for grasp %d of %d' %(grasp.grasp_id, len(grasps)))
            if grasp.grasp_id not in grasp_metrics.keys():
                grasp_metrics[grasp.grasp_id] = {}

            # compute ferrari canny
            if 'ferrari_canny_L1' in config['deterministic_metrics']:
                grasp_metrics[grasp.grasp_id]['ferrari_canny_L1'] = \
                    quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'ferrari_canny_L1', friction_coef=config['friction_coef'],
                                                              num_cone_faces=config['num_cone_faces'], soft_fingers=True)

            # compute force closure
            if 'force_closure' in config['deterministic_metrics']:
                grasp_metrics[grasp.grasp_id]['force_closure'] = \
                    quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'force_closure', friction_coef=config['friction_coef'],
                                                              num_cone_faces=config['num_cone_faces'], soft_fingers=True)

            
            # compute partial closure for each stable pose
            if 'partial_closure' in config['deterministic_metrics']:
                for stable_pose, stable_pose_wrench in zip(stable_poses, stable_pose_wrenches):
                    pc_tag = 'lift_closure_%s' %(stable_pose.id)
                    params = {}
                    params['force_limits'] = grasp_force_limit
                    params['target_wrench'] = stable_pose_wrench
                    grasp_metrics[grasp.grasp_id][pc_tag] = \
                        quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'partial_closure', friction_coef=config['friction_coef'],
                                                                  num_cone_faces=config['num_cone_faces'], soft_fingers=True, params=params)
           
        # compute robust quality metrics
        uncertainty_configs = [low_u_config, med_u_config, high_u_config]
        logging.info('Computing robust quality')

        # iterate through levels of uncertainty
        for j, config in enumerate(uncertainty_configs):
            graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, config)
            f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])

            pfc_tag = db.generate_metric_tag('pfc', config)
            vfc_tag = db.generate_metric_tag('vfc', config)
            efcny_tag = db.generate_metric_tag('efcny_L1', config)

            # iterate through grasps
            for i, grasp in enumerate(grasps):
                logging.info('Evaluating robustness for grasp %d of %d using config %d' %(grasp.grasp_id, len(grasps), j))
                grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, config)
                if grasp.grasp_id not in grasp_metrics.keys():
                    grasp_metrics[grasp.grasp_id] = {}

                # probability of force closure
                logging.info('Computing probability of force closure')
                if 'force_closure' in config['robust_metrics']:
                    pfc, vfc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric='force_closure',
                                                                          num_samples=config['pfc_num_samples'], compute_variance=True)
                    grasp_metrics[grasp.grasp_id][pfc_tag] = pfc
                    grasp_metrics[grasp.grasp_id][vfc_tag] = vfc

                #probability of partial closure
                logging.info("Computing probability of partial closure")
                if 'partial_closure' in config['robust_metrics']:
                    for stable_pose, wrench in zip(stable_poses, stable_pose_wrenches):
                        params = {"force_limits": grasp_force_limit, "target_wrench": wrench}
                        params_rv = rvs.ArtificialSingleRV(params)
                        ppc, vpc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric='partial_closure',
                                                                              params_rv=params_rv, num_samples=config['ppc_num_samples'], compute_variance=True)
                    
                        ppc_tag = db.generate_metric_tag('ppc_%s' %(stable_pose.id), config)
                        vpc_tag = db.generate_metric_tag('vpc_%s' %(stable_pose.id), config)
                                                                      
                        grasp_metrics[grasp.grasp_id][ppc_tag] = ppc
                        grasp_metrics[grasp.grasp_id][vpc_tag] = vpc
                
                # expected ferrari canny
                logging.info('Computing expected ferrari canny')
                if 'ferrari_canny_L1' in config['robust_metrics']:
                    eq = rgq.RobustGraspQuality.expected_quality(graspable_rv, grasp_rv, f_rv, config, quality_metric='ferrari_canny_L1',
                                                                 num_samples=config['eq_num_samples'])
                    grasp_metrics[grasp.grasp_id][efcny_tag] = eq

        quality_stop_time = time.time()
        logging.info('Quality computation for %d grasps took %f sec.' %(len(grasps), quality_stop_time - quality_start_time))

        # store grasp metrics
        output_ds.store_grasp_metrics(obj.key, grasp_metrics, gripper=gripper_name, force_overwrite=True)

    # report total time
    stop_time = time.time()
    logging.info('Labelling for %d grasps took %f sec in total.' %(len(grasps), stop_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dest')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    database_filename = os.path.join(config['database_dir'], config['database_name'])
    if config['write_in_place']:
        logging.info('Writing back to original dataset')
        database = db.Hdf5Database(database_filename, config, access_level=db.READ_WRITE_ACCESS)
    else:
        logging.info('Writing to separate db')
        database = db.Hdf5Database(database_filename, config)
        output_db_filename = os.path.join(args.output_dest, config['results_database_name'])
        output_db = db.Hdf5Database(output_db_filename, config, access_level=db.WRITE_ACCESS)

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # make dataset output directory
        if config['write_in_place']:
            output_ds = dataset
        else:
            dest = os.path.join(args.output_dest, dataset.name)
            try:
                os.makedirs(dest)
            except os.error:
                pass
            output_ds = output_db.create_dataset(dataset_name)

        # label each object in the dataset with grasps
        for obj in dataset:
            if not config['write_in_place']:            
                output_ds.create_graspable(obj.key)

            for gripper_name in config['grippers']:
                logging.info('Labelling object {} with grasps for gripper {}'.format(obj.key, gripper_name))
                grasps = dataset.grasps(obj.key, gripper=gripper_name)

                # create the graspable if necessary
                if config['delete_existing_grasps']:
                    logging.warning('Deleting existing grasps for object {}'.format(obj.key))
                    output_ds.delete_grasps(obj.key, gripper=gripper_name)
                elif config['skip_labeled_objects'] and len(grasps) > 0:
                    logging.info('Object %s already has grasps. Skipping...' %(obj.key)) 
                    continue

                try:
                    label_grasps(obj, dataset, output_ds, gripper_name, config)
                except Exception as e:
                    logging.warning('Failed to complete grasp labelling for object {}'.format(obj.key))
                    logging.warning(str(e))

    database.close()

    if not config['write_in_place']:
        output_db.close()
