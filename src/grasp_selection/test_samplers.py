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
import prior_computation_engine as pce


def run_ua_on(obj, config):
    # sample grasps
    sample_start = time.clock()
    if config['grasp_sampler'] == 'antipodal':
        logging.info('Using antipodal grasp sampling')
        sampler = ags.AntipodalGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

        # pad with gaussian grasps
        num_grasps = len(grasps)
        min_num_grasps = config['min_num_grasps']
        if num_grasps < min_num_grasps:
            target_num_grasps = min_num_grasps - num_grasps
            gaussian_sampler = gs.GaussianGraspSampler(config)
            gaussian_grasps = gaussian_sampler.generate_grasps(
                obj, target_num_grasps=target_num_grasps, check_collisions=config['check_collisions'])
            grasps.extend(gaussian_grasps)
    else:
        logging.info('Using Gaussian grasp sampling')
        sampler = gs.GaussianGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

    # generate pfc candidates
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
    candidates = []
    for grasp in grasps:
        logging.info('Adding grasp %d candidate' %(len(candidates)))
        grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
        pfc_rv = pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config)
        candidates.append(pfc_rv)
    logging.info('%d candidates', len(candidates))

    brute_force_iter = config['bandit_brute_force_iter']*len(candidates)
    snapshot_rate = config['bandit_snapshot_rate']
    objective = objectives.RandomBinaryObjective()

    ua = das.UniformAllocationMean(objective, candidates)
    logging.info('Running uniform allocation for true pfc.')
    bandit_start = time.clock()
    ua_result = ua.solve(
        termination_condition=tc.MaxIterTerminationCondition(brute_force_iter),
        snapshot_rate=snapshot_rate)
    bandit_end = time.clock()
    bandit_duration = bandit_end - bandit_start
    logging.info('Uniform allocation (%d iters) took %f sec' %(brute_force_iter, bandit_duration))

    return ua_result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/correlated.yaml')
    parser.add_argument('output_dest', default='out/')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    chunk = db.Chunk(config)

    # make output directory
    dest = os.path.join(args.output_dest, chunk.name)
    try:
        os.makedirs(dest)
    except os.error:
        pass

    # loop through objects, labelling each
    results = []
    for obj in chunk:
        ua_result = run_ua_on(obj, config)

        logging.info('Saving results to %s' %(dest))
        results = {
            'obj': obj.key,
            'max_pfc': ua_result.best_pred_means[0],
            'num_grasps': config['min_num_grasps'],
            'grasp_sampler': config['grasp_sampler'],
            'ua_result': ua_result
        }
        out_filename = os.path.join(dest, 'results.pkl')
        with open(out_filename, 'w') as f:
            pkl.dump(results, f)


