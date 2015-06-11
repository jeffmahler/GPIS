"""
Main file for labelling objects with grasps (local now, will be distributed in future)

Author: Jeff Mahler
"""
import json
import IPython
import logging
import numpy as np
import os
import sys
import time

import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import objectives
import pfc
import pr2_grasp_checker as pgc
import termination_conditions as tc

GRASP_SAVE_PATH = '/home/jmahler/tmp_grasps'

def label_pfc(obj, dataset, config):
    """ Label an object with grasps according to probability of force closure """
    # sample intial antipodal grasps
    start = time.clock()
    sampler = ags.AntipodalGraspSampler(config)
        
    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(obj, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    # partition grasps
    grasp_partitions = pfc.space_partition_grasps(grasps, config)

    # bandit params
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    tc_list = [tc.MaxIterTerminationCondition(max_iter), tc.ConfidenceTerminationCondition(confidence)]

    # run bandits on each partition
    object_grasps = []
    grasp_qualities = []
    i = 0
    for grasp_partition in grasp_partitions:
        logging.info('Finding highest quality grasp in partition %d' %(i))
        # create random variables
        graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
        f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction gaussian random variable
        candidates = []

        for grasp in grasp_partition:
            grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
            candidates.append(pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config))

        # run bandits
        objective = objectives.RandomBinaryObjective()
        ts = das.ThompsonSampling(objective, candidates)
        ts_result = ts.solve(termination_condition = tc.OrTerminationCondition(tc_list), snapshot_rate = snapshot_rate)
        object_grasps.extend([c.grasp for c in ts_result.best_candidates])
        grasp_qualities.extend(list(ts_result.best_pred_means))
        i = i+1

    stop = time.clock()
    logging.info('Took %d sec' %(stop - start))

    # get rotated, translated versions of grasps
    delay = 0
    pr2_grasps = []
    pr2_grasp_qualities = []
    theta_res = config['grasp_theta_res'] * np.pi
    grasp_checker = pgc.OpenRaveGraspChecker(view=config['vis_grasps'])
    i = 0
    if config['vis_grasps']:
        delay = config['vis_delay']

    for grasp in object_grasps:
        rotated_grasps = grasp.transform(obj.tf, theta_res)
        rotated_grasps = grasp_checker.prune_grasps_in_collision(obj, rotated_grasps, auto_step=True, close_fingers=False, delay=delay) 
        pr2_grasps.extend(rotated_grasps)
        pr2_grasp_qualities.extend([grasp_qualities[i]] * len(rotated_grasps))
        i = i+1                    

    logging.info('Num grasps: %d' %(len(pr2_grasps)))

    # save grasps locally :( Due to problems with sudo
    grasp_filename = os.path.join(GRASP_SAVE_PATH, obj.key + '.json')
    with open(grasp_filename, 'w') as f:
        json.dump([pr2_grasps[i].to_json(quality=pr2_grasp_qualities[i]) for i in range(len(pr2_grasps))], f,
                  sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/basic_labelling.yaml')
    parser.add_argument('output_dest', default='out/')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    chunk = db.Chunk(config)

    # make output directory
    dest = os.path.join(args.output_dest, chunk.name)
    GRASP_SAVE_PATH = dest # legacy
    try:
        os.makedirs(dest)
    except os.error:
        pass

    # loop through objects, labelling each
    for obj in chunk:
        logging.info('Labelling object {}'.format(obj.key))
        label_pfc(obj, chunk, config)
