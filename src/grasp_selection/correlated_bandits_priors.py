"""
Main file for correlated bandit experiments.

Author: Brian Hou
"""
import sys
sys.path.insert(0, 'src/grasp_selection/feature_vectors/')
import feature_database

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

import scipy.spatial.distance as ssd

from correlated_bandits import experiment_hash, reward_vs_iters

class BanditCorrelatedExperimentResult:
    def __init__(self, ua_reward, ts_reward, ts_corr_reward, ts_corr_prior_reward,
                 true_avg_reward, iters, kernel_matrix,
                 obj_key='', num_objects=1):
        self.ua_reward = ua_reward
        self.ts_reward = ts_reward
        self.ts_corr_reward = ts_corr_reward
        self.ts_corr_prior_reward = ts_corr_prior_reward
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
        ts_corr_prior_rewards = []
        for x in range(0, len(result_list[0].ts_corr_prior_reward)):
            ts_corr_prior_rewards.append(np.zeros([len(result_list), result_list[0].ts_corr_reward.shape[0]]))

        i = 0
        obj_keys = []
        for r in result_list:
            ua_reward[i,:] = r.ua_reward
            ts_reward[i,:] = r.ts_reward
            ts_corr_reward[i,:] = r.ts_corr_reward
            for n, ts_corr_prior_reward in enumerate(ts_corr_prior_rewards):
                ts_corr_prior_reward[i,:] = r.ts_corr_prior_reward[n]
            obj_keys.append(r.obj_key)
            i = i + 1

        true_avg_rewards = [r.true_avg_reward for r in result_list]
        iters = [r.iters for r in result_list]
        kernel_matrices = [r.kernel_matrix for r in result_list]

        return BanditCorrelatedExperimentResult(ua_reward, ts_reward, ts_corr_reward, ts_corr_prior_rewards,
                                                true_avg_rewards,
                                                iters,
                                                kernel_matrices,
                                                obj_keys,
                                                len(result_list))

def label_correlated(obj, chunk, config, plot=False,
                     priors_dataset=None, nearest_features_names=None):
    """Label an object with grasps according to probability of force closure,
    using correlated bandits."""
    bandit_start = time.clock()

    np.random.seed(100)

    # load grasps from database
    sample_start = time.clock()
    grasps = chunk.load_grasps(obj.key)
    sample_end = time.clock()
    sample_duration = sample_end - sample_start
    logging.info('Loaded %d grasps' %(len(grasps)))
    logging.info('Grasp candidate loading took %f sec' %(sample_duration))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return None

    # load features for all grasps
    feature_start = time.clock()
    feature_loader = ff.GraspableFeatureLoader(obj, chunk.name, config)
    all_features = feature_loader.load_all_features(grasps) # in same order as grasps
    feature_end = time.clock()
    feature_duration = feature_end - feature_start
    logging.info('Loaded %d features' %(len(all_features)))
    logging.info('Grasp feature loading took %f sec' %(feature_duration))

    # bandit params
    num_trials = config['num_trials']
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]

    # run bandits!
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

    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
    objective = objectives.RandomBinaryObjective()

    # compute priors
    if priors_dataset is None:
        priors_dataset = chunk
    prior_engine = pce.PriorComputationEngine(priors_dataset, config)

    all_alpha_priors = []
    all_beta_priors = []
    if nearest_features_names == None:
        alpha_priors, beta_priors = prior_engine.compute_priors(obj, candidates)
        all_alpha_priors.append(alpha_priors)
        all_beta_priors.append(beta_priors)
    else:
        for nearest_features_name in nearest_features_names:
            alpha_priors, beta_priors = prior_engine.compute_priors(obj, candidates, nearest_features_name=nearest_features_name)
            all_alpha_priors.append(alpha_priors)
            all_beta_priors.append(beta_priors)

    # pre-computed pfc values
    estimated_pfc = np.array([c.grasp.quality for c in candidates])

    # run bandits for several trials
    ua_rewards = []
    ts_rewards = []
    ts_corr_rewards = []
    all_ts_corr_prior_rewards = []
    for x in range(0, len(all_alpha_priors)):
        all_ts_corr_prior_rewards.append([])

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

        # correlated Thompson sampling with priors for even faster than that convergence
        for alpha_priors, beta_priors, ts_corr_prior_rewards in zip(all_alpha_priors, all_beta_priors, all_ts_corr_prior_rewards):
            ts_corr_prior = das.CorrelatedThompsonSampling(
                objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'], alpha_prior = alpha_priors, beta_prior = beta_priors)
            logging.info('Running correlated Thompson sampling with priors.')
            ts_corr_prior_result = ts_corr_prior.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)
            ts_corr_prior_normalized_reward = reward_vs_iters(ts_corr_prior_result, estimated_pfc)
            ts_corr_prior_rewards.append(ts_corr_prior_normalized_reward)

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

    all_avg_ts_corr_prior_rewards = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        all_avg_ts_corr_prior_rewards.append(np.mean(np.array(ts_corr_prior_rewards), axis=0))

    # compute avg normalized rewards
    avg_ua_rewards = np.mean(all_ua_rewards, axis=0)
    avg_ts_rewards = np.mean(all_ts_rewards, axis=0)
    avg_ts_corr_rewards = np.mean(all_ts_corr_rewards, axis=0)

    # kernel matrix
    kernel_matrix = kernel.matrix(candidates)

    return BanditCorrelatedExperimentResult(avg_ua_rewards, avg_ts_rewards, avg_ts_corr_rewards,
                                            all_avg_ts_corr_prior_rewards,
                                            estimated_pfc, ua_result.iters, kernel_matrix, obj_key=obj.key)

def plot_kernels_for_key(obj, chunk, config, priors_dataset=None, nearest_features_name=None):
    # load grasps from database
    grasps = chunk.load_grasps(obj.key)
    logging.info('Loaded %d grasps' %(len(grasps)))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return None

    # load features for all grasps
    feature_start = time.clock()
    feature_loader = ff.GraspableFeatureLoader(obj, chunk.name, config)
    all_features = feature_loader.load_all_features(grasps) # in same order as grasps
    feature_end = time.clock()
    feature_duration = feature_end - feature_start
    logging.info('Loaded %d features' %(len(all_features)))
    logging.info('Grasp feature loading took %f sec' %(feature_duration))

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

    if priors_dataset is None:
        priors_dataset = chunk
    prior_engine = pce.PriorComputationEngine(priors_dataset, config)
    if nearest_features_name == None:
        neighbor_keys, all_neighbor_kernels, all_neighbor_pfc_diffs, all_distances = prior_engine.compute_grasp_kernels(obj, candidates)
    else:
        neighbor_keys, all_neighbor_kernels, all_neighbor_pfc_diffs, all_distances = prior_engine.compute_grasp_kernels(obj, candidates, nearest_features_name=nearest_features_name)


    for neighbor_key, object_distance in zip(neighbor_keys, all_distances):
        print '%s and %s: %.5f' % (obj.key, neighbor_key, object_distance)

    # feature transform
    def phi(rv):
        return rv.features
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
    estimated_pfc = np.array([c.grasp.quality for c in candidates])

    k = kernel.matrix(candidates)
    k_vec = k.ravel()
    pfc_arr = np.array([estimated_pfc]).T
    pfc_diff = ssd.squareform(ssd.pdist(pfc_arr))
    pfc_vec = pfc_diff.ravel()

    bad_ind = np.where(pfc_diff > 1.0 - k) 

    labels = [obj.key[:15]] + map(lambda x: x[:15], neighbor_keys)
    scatter_objs =[]
    plt.figure()
    colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(all_neighbor_pfc_diffs)))
    scatter_objs.append(plt.scatter(k_vec, pfc_vec, c='#eeeeff'))
    for i, (neighbor_pfc_diffs, neighbor_kernels) in enumerate(zip(all_neighbor_pfc_diffs, all_neighbor_kernels)):
        scatter_objs.append(plt.scatter(neighbor_kernels, neighbor_pfc_diffs, c=colors[i]))
    plt.xlabel('Kernel', fontsize=15)
    plt.ylabel('PFC Diff', fontsize=15)
    plt.title('Correlations', fontsize=15)
    plt.legend(scatter_objs, labels)

def plot_prior_diffs(obj, chunk, config, nearest_features_name=None):
    # load grasps from database
    sample_start = time.clock()
    grasps = chunk.load_grasps(obj.key)
    sample_end = time.clock()
    sample_duration = sample_end - sample_start
    logging.info('Loaded %d grasps' %(len(grasps)))
    logging.info('Grasp candidate loading took %f sec' %(sample_duration))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return None

    # run bandits!
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    # load features for all grasps
    feature_start = time.clock()
    feature_loader = ff.GraspableFeatureLoader(obj, chunk.name, config)
    all_features = feature_loader.load_all_features(grasps) # in same order as grasps
    feature_end = time.clock()
    feature_duration = feature_end - feature_start
    logging.info('Loaded %d features' %(len(all_features)))
    logging.info('Grasp feature loading took %f sec' %(feature_duration))

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

    prior_engine = pce.PriorComputationEngine(chunk, config)
    alpha_priors, beta_priors = prior_engine.compute_priors(obj, candidates, nearest_features_name=nearest_features_name)

    prior_means = models.BetaBernoulliModel.beta_mean(np.array(alpha_priors), np.array(beta_priors))
    prior_variances = models.BetaBernoulliModel.beta_variance(np.array(alpha_priors), np.array(beta_priors))

    diffs = []
    variances = []
    for i, candidate in enumerate(candidates):
        diffs.append(abs(prior_means[i] - candidate.grasp.quality))
        variances.append(prior_variances[i])

    plt.figure()
    plt.scatter(variances, diffs, c=u'b')
    plt.xlim(0, 0.1)
    plt.ylim(0, 1)
    plt.xlabel('Variance', fontsize=15)
    plt.ylabel('Error', fontsize=15)

def run_and_save_experiment(obj, chunk, priors_dataset, config, result_dir):
    nearest_features_names = ['nearest_features_15', 'nearest_features_150', 'nearest_features_all']
    # plot params
    line_width = config['line_width']
    font_size = config['font_size']
    dpi = config['dpi']

    results = []
    avg_experiment_result = None
    experiment_result = label_correlated(obj, chunk, config,
                                         priors_dataset=priors_dataset,
                                         nearest_features_names=nearest_features_names)
    results.append(experiment_result)

    if len(results) == 0:
        logging.info('Exiting. No grasps found')
        exit(0)

    # combine results
    all_results = BanditCorrelatedExperimentResult.compile_results(results)

    logging.info('Creating and saving plots...')

    # plotting of final results
    ua_normalized_reward = np.mean(all_results.ua_reward, axis=0)
    ts_normalized_reward = np.mean(all_results.ts_reward, axis=0)
    ts_corr_normalized_reward = np.mean(all_results.ts_corr_reward, axis=0)

    all_ts_corr_prior_rewards = all_results.ts_corr_prior_reward
    ts_corr_prior_normalized_reward = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        ts_corr_prior_normalized_reward.append(np.mean(ts_corr_prior_rewards, axis=0))

    plt.figure()
    ua_obj = plt.plot(all_results.iters[0], ua_normalized_reward,
                      c=u'k', linewidth=2.0, label='Uniform')
    ts_obj = plt.plot(all_results.iters[0], ts_normalized_reward,
                      c=u'g', linewidth=2.0, label='TS (Uncorrelated)')
    ts_corr_obj = plt.plot(all_results.iters[0], ts_corr_normalized_reward,
                      c=u'r', linewidth=2.0, label='TS (Correlated)')
    ts_corr_prior_0_obj = plt.plot(all_results.iters[0], ts_corr_prior_normalized_reward[0],
                      c=u'c', linewidth=2.0, label='TS (Priors_15)')
    ts_corr_prior_1_obj = plt.plot(all_results.iters[0], ts_corr_prior_normalized_reward[1],
                      c=u'm', linewidth=2.0, label='TS (Priors_150)')
    ts_corr_prior_2_obj = plt.plot(all_results.iters[0], ts_corr_prior_normalized_reward[2],
                      c=u'b', linewidth=2.0, label='TS (Priors_all)')
    plt.xlim(0, np.max(all_results.iters[0]))
    plt.ylim(0.5, 1)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(result_dir, obj.key+'_results.png'), dpi=dpi, bbox_extra_artists=(legend,), bbox_inches='tight')

    # plot kernels
    plot_kernels_for_key(obj, chunk, config, priors_dataset, nearest_features_names[0])
    plt.savefig(os.path.join(result_dir, obj.key+'_kernels_15.png'), dpi=dpi)

    plot_kernels_for_key(obj, chunk, config, priors_dataset, nearest_features_names[1])
    plt.savefig(os.path.join(result_dir, obj.key+'_kernels_150.png'), dpi=dpi)

    plot_kernels_for_key(obj, chunk, config, priors_dataset, nearest_features_names[3])
    plt.savefig(os.path.join(result_dir, obj.key+'_kernels_all.png'), dpi=dpi)

    # plot histograms
    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins+1)
    plt.figure()
    n, bins, patches = plt.hist(experiment_result.true_avg_reward, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)
    plt.savefig(os.path.join(result_dir,  obj.key+'_histogram.png'), dpi=dpi)

    # plot_prior_diffs(obj, chunk, config, nearest_features_name=nearest_features_names[2])
    # plt.savefig(os.path.join(result_dir, obj.key+'_errors_all.png'), dpi=dpi)

    # # save to file
    # logging.info('Saving results to %s' %(dest))
    # for r in results:
    #     r.save(dest)

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

    if 'priors_dataset' in config:
        priors_dataset = db.Dataset(config['priors_dataset'], config)
    else:
        priors_dataset = None

    # make output directory
    dest = os.path.join(args.output_dest, chunk.name)
    try:
        os.makedirs(dest)
    except os.error:
        pass

    for obj in chunk:
        run_and_save_experiment(obj, chunk, priors_dataset, config, dest)
