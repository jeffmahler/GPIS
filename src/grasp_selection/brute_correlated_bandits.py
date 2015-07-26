B"""
Main file for correlated bandit experiments.

Author: Brian Hou
"""
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import json_serialization as jsons
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp as g
import grasp_sampler as gs
import json_serialization as jsons
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import termination_conditions as tc

def experiment_hash(N = 10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def reward_vs_iters(result, true_pfc, normalize=True):
    """Computes the expected values for the best arms of a BetaBernoulliModel at
    each time step.
    Params:
        result - AdaptiveSamplingResult instance, from a BetaBernoulliModel
        normalize - Divide by true best value
    Returns:
        best_values - list of floats, expected values over time
    """
    true_best_value = np.max(true_pfc)
    best_pred_values = [true_pfc[m.best_pred_ind] for m in result.models]
    if normalize:
        best_pred_values = best_pred_values / true_best_value

    return best_pred_values

def save_grasps(grasps, pfcs, obj, dest, num_successes=0, num_failures=0):
    """
    Save grasps as json files with indices
    """
    i = 0
    for grasp, pfc in zip(grasps, pfcs):
        grasp_json = grasp.to_json(quality=pfc, method='PFC', num_successes=num_successes, num_failures=num_failures)
        grasp_filename = os.path.join(dest, obj.key + '_' + str(i) + '.json')
        with open(grasp_filename, 'w') as grasp_file:
            jsons.dump(grasp_json, grasp_file)
        i += 1

class BanditCorrelatedExperimentResult:
    """ Encapsulates useful info for experiment results"""
    def __init__(self, ua_reward, ts_reward, ts_corr_reward, true_avg_reward, iters, kernel_matrix, obj_key = '', num_objects = 1):
        self.ua_reward = ua_reward
        self.ts_reward = ts_reward
        self.ts_corr_reward = ts_corr_reward
        self.true_avg_reward = true_avg_reward

        self.iters = iters
        self.kernel_matrix = kernel_matrix

        self.obj_key = obj_key
        self.num_objects = num_objects

    def save(self, out_dir):
        """ Save this object to a pickle file in specified dir """
        out_filename = os.path.join(out_dir, self.obj_key + '.pkl')
        with open(out_filename, 'w') as f:
            pkl.dump(self, f)

    @staticmethod
    def compile_results(result_list):
        """ Put all results in a giant list """
        if len(result_list) == 0:
            return None

        ua_reward = np.zeros([len(result_list), result_list[0].ua_reward.shape[0]])
        ts_reward = np.zeros([len(result_list), result_list[0].ts_reward.shape[0]])
        ts_corr_reward = np.zeros([len(result_list), result_list[0].ts_corr_reward.shape[0]])

        i = 0
        obj_keys = []
        for r in result_list:
            ua_reward[i,:] = r.ua_reward
            ts_reward[i,:] = r.ts_reward
            ts_corr_reward[i,:] = r.ts_corr_reward
            obj_keys.append(r.obj_key)
            i = i + 1

        true_avg_rewards = [r.true_avg_reward for r in result_list]
        iters = [r.iters for r in result_list]
        kernel_matrices = [r.kernel_matrix for r in result_list]

        return BanditCorrelatedExperimentResult(ua_reward, ts_reward, ts_corr_reward,
                                                true_avg_rewards,
                                                iters,
                                                kernel_matrices,
                                                len(result_list))

def label_correlated(obj, chunk, dest, config, plot=False, load=True):
    """Label an object with grasps according to probability of force closure,
    using correlated bandits."""
    bandit_start = time.clock()
    #np.random.seed(100)

    # sample grasps
    sample_start = time.clock()
                              
    if config['grasp_sampler'] == 'antipodal':
        logging.info('Using antipodal grasp sampling')
        sampler = ags.AntipodalGraspSampler(config)
        grasps = sampler.generate_grasps(obj, check_collisions=config['check_collisions'], vis=False)

        # pad with gaussian grasps
        num_grasps = len(grasps)
        min_num_grasps = config['min_num_grasps']
        if num_grasps < min_num_grasps:
            target_num_grasps = min_num_grasps - num_grasps
            gaussian_sampler = gs.GaussianGraspSampler(config)        
            gaussian_grasps = gaussian_sampler.generate_grasps(obj, target_num_grasps=target_num_grasps,
                                                                   check_collisions=config['check_collisions'], vis=plot)
            grasps.extend(gaussian_grasps)
    else:
        logging.info('Using Gaussian grasp sampling')
        sampler = gs.GaussianGraspSampler(config)        
        grasps = sampler.generate_grasps(obj, check_collisions=config['check_collisions'], vis=plot,
                                             grasp_gen_mult = 6)
    sample_end = time.clock()
    sample_duration = sample_end - sample_start
    logging.info('Loaded %d grasps' %(len(grasps)))
    logging.info('Grasp candidate loading took %f sec' %(sample_duration))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return None

    # extract load features for all grasps
    feature_start = time.clock()
    feature_extractor = ff.GraspableFeatureExtractor(obj, config)
    all_features = feature_extractor.compute_all_features(grasps)
    feature_end = time.clock()
    feature_duration = feature_end - feature_start
    logging.info('Loaded %d features' %(len(all_features)))
    logging.info('Grasp feature loading took %f sec' %(feature_duration))

    # bandit params
    num_trials = config['num_trials']
    brute_force_iter = config['bandit_brute_force_iter']
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    brute_snapshot_rate = config['bandit_brute_force_snapshot_rate']
    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]

    # set up randome variables
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    candidates = []
    for grasp, features in zip(grasps, all_features):
        logging.info('Adding grasp %d' %len(candidates))
        grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
        pfc_rv = pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config)
        if features is None:
            logging.info('Could not compute features for grasp.')
        else:
            pfc_rv.set_features(features)
            candidates.append(pfc_rv)

    # feature transform
    def phi(rv):
        return rv.features

    # create nn structs for kernels
    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
    objective = objectives.RandomBinaryObjective()

    # uniform allocation for true values
    ua_brute = das.UniformAllocationMean(objective, candidates)
    logging.info('Running uniform allocation for true pfc.')
    ua_brute_result = ua_brute.solve(termination_condition=tc.MaxIterTerminationCondition(brute_force_iter),
                                     snapshot_rate=brute_snapshot_rate)
    final_model = ua_brute_result.models[-1]
    estimated_pfc = models.BetaBernoulliModel.beta_mean(final_model.alphas, final_model.betas)
    save_grasps(grasps, estimated_pfc, obj, dest, num_successes=final_model.alphas, num_failures=final_model.betas)

    # run bandits for several trials
    ua_rewards = []
    ts_rewards = []
    ts_corr_rewards = []

    for t in range(num_trials):
        logging.info('Trial %d' %(t))

        # Uniform sampling
        ua = das.UniformAllocationMean(objective, candidates)
        logging.info('Running Uniform allocation.')
        ua_result = ua.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # Thompson sampling
        ts = das.ThompsonSampling(objective, candidates)
        logging.info('Running Thompson sampling.')
        ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # correlated Thompson sampling for even faster convergence
        ts_corr = das.CorrelatedThompsonSampling(
            objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'])
        logging.info('Running correlated Thompson sampling.')
        ts_corr_result = ts_corr.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # compile results
        ua_normalized_reward = reward_vs_iters(ua_result, estimated_pfc)
        ts_normalized_reward = reward_vs_iters(ts_result, estimated_pfc)
        ts_corr_normalized_reward = reward_vs_iters(ts_corr_result, estimated_pfc)
        
        ua_rewards.append(ua_normalized_reward)
        ts_rewards.append(ts_normalized_reward)
        ts_corr_rewards.append(ts_corr_normalized_reward)

    # get the bandit rewards
    all_ua_rewards = np.array(ua_rewards)
    all_ts_rewards = np.array(ts_rewards)
    all_ts_corr_rewards = np.array(ts_corr_rewards)

    # compute avg normalized rewards
    avg_ua_rewards = np.mean(all_ua_rewards, axis=0)
    avg_ts_rewards = np.mean(all_ts_rewards, axis=0)
    avg_ts_corr_rewards = np.mean(all_ts_corr_rewards, axis=0)

    # kernel matrix
    kernel_matrix = kernel.matrix(candidates)

    return BanditCorrelatedExperimentResult(avg_ua_rewards, avg_ts_rewards, avg_ts_corr_rewards,
                                            estimated_pfc, ua_result.iters, kernel_matrix, obj_key=obj.key)
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
    avg_experiment_result = None
    for obj in chunk:
        logging.info('Labelling object {}'.format(obj.key))
        experiment_result = label_correlated(obj, chunk, dest, config)
        if experiment_result is None:
            continue # no grasps to run bandits on for this object
        results.append(experiment_result)

    if len(results) == 0:
        logging.info('Exiting. No grasps found')
        exit(0)

    # save each result to a file
    logging.info('Saving results to %s' %(dest))
    for r in results:
        r.save(dest)
