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
    sampler = ags.AntipodalGraspSampler(config)
        
    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(obj, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    # partition grasps
    grasp_partitions = pfc.space_partition_grasps(grasps, config)

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
        ts_result = ts.solve(termination_condition = tc.MaxIterTerminationCondition(200), snapshot_rate = 100)
        object_grasps.extend([c.grasp for c in ts_result.best_candidates])
        grasp_qualities.extend(list(ts_result.best_pred_means))
        i = i+1
        

#    for grasp in object_grasps:
#        grasp.close_fingers(obj, vis=True)
#        time.sleep(1)

    # get rotated, translated versions of grasps
    pr2_grasps = []
    pr2_grasp_qualities = []
    theta_res = config['grasp_theta_res'] * np.pi
    grasp_checker = pgc.OpenRaveGraspChecker()
    i = 0
    for grasp in object_grasps:
        rotated_grasps = grasp.transform(obj.tf, theta_res)
        rotated_grasps = grasp_checker.prune_grasps_in_collision(obj, rotated_grasps, auto_step=True, close_fingers=False, delay = 1) 
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
    logging.getLogger().setLevel(logging.INFO)

    # get command line args
    argc = len(sys.argv)
    config_filename = sys.argv[1]

    # read config file
    config = ec.ExperimentConfig(config_filename)

    # loop through objects, labelling each
    database = db.Database(config)
    for dataset in database.datasets:
        logging.info('Labelling dataset %s' %(dataset.name))
        for obj in dataset:
#            obj = dataset[2]#['elmers_washable_no_run_school_glue']
            logging.info('Labelling object %s' %(obj.key))
            label_pfc(obj, dataset, config)
