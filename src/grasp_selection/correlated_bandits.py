"""
Main file for correlated bandit experiments.

Author: Brian Hou
"""
import json
import logging
import pickle as pkl
import os
import time

import random
import string

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import termination_conditions as tc

def experiment_hash(N = 10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

class BanditCorrelatedExperimentResult:
    def __init__(self, ua_reward, ts_reward, ts_corr_reward, ua_result, ts_result, ts_corr_result, obj_key = '', num_objects = 1):
        self.ua_reward = ua_reward
        self.ts_reward = ts_reward
        self.ts_corr_reward = ts_corr_reward

        self.ua_result = ua_result
        self.ts_result = ts_result
        self.ts_corr_result = ts_corr_result

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

        ua_results = [r.ua_result for r in result_list]
        ts_results = [r.ts_result for r in result_list]
        ts_corr_results = [r.ts_corr_result for r in result_list]

        return BanditCorrelatedExperimentResult(ua_reward, ts_reward, ts_corr_reward,
                                                ua_results,
                                                ts_results,
                                                ts_corr_results,
                                                obj_keys,
                                                len(result_list))

def reward_vs_iters(result, true_pfc, plot=False, normalize=True):
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

    if plot:
        plt.figure()
        plt.plot(result.iters, best_pred_values, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('P(Success)')

    return best_pred_values

def label_correlated(obj, dest, config, plot=False):
    """Label an object with grasps according to probability of force closure,
    using correlated bandits."""
    bandit_start = time.clock()

    np.random.seed(100)

    # sample initial antipodal grasps
    sampler = ags.AntipodalGraspSampler(config)
    antipodal_start = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(obj, check_collisions=config['check_collisions'], vis=False)
    antipodal_end = time.clock()
    antipodal_duration = antipodal_end - antipodal_start
    logging.info('Antipodal grasp candidate generation took %f sec' %(antipodal_duration))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return None

    # bandit params
    brute_force_iter = config['bandit_brute_force_iter']
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
#        tc.ConfidenceTerminationCondition(confidence)
    ]

    # run bandits!
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    # compute feature vectors for all grasps
    feature_extractor = ff.GraspableFeatureExtractor(obj, config)
    all_features = feature_extractor.compute_all_features(grasps)

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

    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
    objective = objectives.RandomBinaryObjective()

    # uniform allocation for true values
    ua = das.UniformAllocationMean(objective, candidates)
    logging.info('Running uniform allocation for true pfc.')
    ua_result = ua.solve(termination_condition=tc.MaxIterTerminationCondition(brute_force_iter),
                         snapshot_rate=snapshot_rate)
    estimated_pfc = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)

    # Thompson sampling for faster convergence
    ts = das.ThompsonSampling(objective, candidates)
    logging.info('Running Thompson sampling.')
    ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

    # correlated Thompson sampling for even faster convergence
    ts_corr = das.CorrelatedThompsonSampling(
        objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'])
    logging.info('Running correlated Thompson sampling.')
    ts_corr_result = ts_corr.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

    object_grasps = [candidates[i].grasp for i in ts_result.best_candidates]
    grasp_qualities = list(ts_result.best_pred_means)

    bandit_stop = time.clock()
    logging.info('Bandits took %f sec' %(bandit_stop - bandit_start))

    # get rotated, translated versions of grasps
    delay = 0
    pr2_grasps = []
    pr2_grasp_qualities = []
    theta_res = config['grasp_theta_res'] * np.pi
#    grasp_checker = pgc.OpenRaveGraspChecker(view=config['vis_grasps'])

    if config['vis_grasps']:
        delay = config['vis_delay']

    for grasp, grasp_quality in zip(object_grasps, grasp_qualities):
        rotated_grasps = grasp.transform(obj.tf, theta_res)
#        rotated_grasps = grasp_checker.prune_grasps_in_collision(obj, rotated_grasps, auto_step=True, close_fingers=False, delay=delay)
        pr2_grasps.extend(rotated_grasps)
        pr2_grasp_qualities.extend([grasp_quality] * len(rotated_grasps))

    logging.info('Num grasps: %d' %(len(pr2_grasps)))

    grasp_filename = os.path.join(dest, obj.key + '.json')
    with open(grasp_filename, 'w') as f:
        json.dump([g.to_json(quality=q) for g, q in zip(pr2_grasps, pr2_grasp_qualities)], f,
                  sort_keys=True, indent=4, separators=(',', ': '))

    ua_normalized_reward = reward_vs_iters(ua_result, estimated_pfc)
    ts_normalized_reward = reward_vs_iters(ts_result, estimated_pfc)
    ts_corr_normalized_reward = reward_vs_iters(ts_corr_result, estimated_pfc)

    return BanditCorrelatedExperimentResult(ua_normalized_reward, ts_normalized_reward, ts_corr_normalized_reward,
                                            ua_result, ts_result, ts_corr_result, obj_key=obj.key)

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
        experiment_result = label_correlated(obj, dest, config)
        if experiment_result is None:
            continue # no grasps to run bandits on for this object
        results.append(experiment_result)

    if len(results) == 0:
        logging.info('Exiting. No grasps found')
        exit(0)

    # combine results
    all_results = BanditCorrelatedExperimentResult.compile_results(results)

    # plotting of final results
    ua_normalized_reward = np.mean(all_results.ua_reward, axis=0)
    ts_normalized_reward = np.mean(all_results.ts_reward, axis=0)
    ts_corr_normalized_reward = np.mean(all_results.ts_corr_reward, axis=0)

    if config['plot']:
        plt.figure()
        ua_obj = plt.plot(all_results.ua_result[0].iters, ua_normalized_reward,
                          c=u'b', linewidth=2.0, label='Uniform Allocation')
        ts_obj = plt.plot(all_results.ts_result[0].iters, ts_normalized_reward,
                          c=u'g', linewidth=2.0, label='Thompson Sampling (Uncorrelated)')
        ts_corr_obj = plt.plot(all_results.ts_corr_result[0].iters, ts_corr_normalized_reward,
                          c=u'r', linewidth=2.0, label='Thompson Sampling (Correlated)')
        plt.xlim(0, np.max(all_results.ts_result[0].iters))
        plt.ylim(0.5, 1)
        plt.legend(loc='lower right')
        plt.show()

    # save to file
    logging.info('Saving results to %s' %(dest))
    for r in results:
        r.save(dest)
