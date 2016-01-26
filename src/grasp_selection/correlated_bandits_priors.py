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
import matplotlib.colors as clr
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
import plotting
import pr2_grasp_checker as pgc
import termination_conditions as tc
import prior_computation_engine as pce

import scipy.spatial.distance as ssd

from correlated_bandits import experiment_hash, reward_vs_iters

# list of keys that cause trouble (right now the ones that take way too damn long)
skip_keys = ['BigBIRD_nutrigrain_strawberry_greek_yogurt', 'ModelNet40_radio_112']

class BanditCorrelatedPriorExperimentResult:
    def __init__(self, ua_reward, ts_reward, gi_reward, ts_corr_reward, bucb_corr_reward, ts_corr_prior_reward, bucb_corr_prior_reward,
                 true_avg_reward, iters, kernel_matrix,
                 neighbor_kernels, neighbor_pfc_diffs, neighbor_distances,
                 ce_vals, se_vals, we_vals, num_grasps, total_weights,
                 ua_ind, ts_ind, ts_corr_ind, bucb_corr_ind, ts_corr_prior_ind, bucb_corr_prior_ind,
                 ua_runtime, ts_runtime, ts_corr_runtime, bucb_corr_runtime, ts_corr_prior_runtime, bucb_corr_prior_runtime, prior_comp_time,
                 obj_key='', neighbor_keys=[], num_objects=1):
        self.ua_reward = ua_reward
        self.ts_reward = ts_reward
        self.gi_reward = gi_reward
        self.ts_corr_reward = ts_corr_reward
        self.bucb_corr_reward = bucb_corr_reward
        self.ts_corr_prior_reward = ts_corr_prior_reward
        self.bucb_corr_prior_reward = bucb_corr_prior_reward
        self.true_avg_reward = true_avg_reward

        self.iters = iters
        self.kernel_matrix = kernel_matrix

        self.neighbor_kernels = neighbor_kernels
        self.neighbor_pfc_diffs = neighbor_pfc_diffs
        self.neighbor_distances = neighbor_distances

        self.ce_vals = ce_vals
        self.se_vals = se_vals
        self.we_vals = we_vals
        self.num_grasps = num_grasps
        self.total_weights = total_weights

        self.ua_ind = ua_ind
        self.ts_ind = ts_ind
        self.ts_corr_ind = ts_corr_ind
        self.bucb_corr_ind = bucb_corr_ind
        self.ts_corr_prior_ind = ts_corr_prior_ind
        self.bucb_corr_prior_ind = bucb_corr_prior_ind

        self.ua_runtime = ua_runtime
        self.ts_runtime = ts_runtime
        self.ts_corr_runtime = ts_corr_runtime
        self.bucb_corr_runtime = bucb_corr_runtime
        self.ts_corr_prior_runtime = ts_corr_prior_runtime
        self.bucb_corr_prior_runtime = bucb_corr_prior_runtime
        self.prior_comp_time = prior_comp_time

        self.obj_key = obj_key
        self.neighbor_keys = neighbor_keys
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
        gi_reward = np.zeros([len(result_list), result_list[0].gi_reward.shape[0]])
        ts_corr_reward = np.zeros([len(result_list), result_list[0].ts_corr_reward.shape[0]])
        bucb_corr_reward = np.zeros([len(result_list), result_list[0].bucb_corr_reward.shape[0]])
        ts_corr_prior_rewards = []
        for x in range(0, len(result_list[0].ts_corr_prior_reward)):
            ts_corr_prior_rewards.append(np.zeros([len(result_list), result_list[0].ts_corr_reward.shape[0]]))
        bucb_corr_prior_rewards = []
        for x in range(0, len(result_list[0].bucb_corr_prior_reward)):
            bucb_corr_prior_rewards.append(np.zeros([len(result_list), result_list[0].bucb_corr_reward.shape[0]]))
            
        ce_vals = np.zeros([len(result_list), result_list[0].ce_vals.shape[0]])
        se_vals = np.zeros([len(result_list), result_list[0].se_vals.shape[0]])
        we_vals = np.zeros([len(result_list), result_list[0].we_vals.shape[0]])
        num_grasps = np.zeros(len(result_list))
        total_weights = np.zeros([len(result_list), result_list[0].total_weights.shape[0]])

        ua_runtime = np.zeros(len(result_list))
        ts_runtime = np.zeros(len(result_list))
        ts_corr_runtime = np.zeros(len(result_list))
        bucb_corr_runtime = np.zeros(len(result_list))
        ts_corr_prior_runtime = np.zeros([len(result_list), len(result_list[0].ts_corr_prior_runtime)])
        bucb_corr_prior_runtime = np.zeros([len(result_list), len(result_list[0].bucb_corr_prior_runtime)])
        prior_comp_time = np.zeros([len(result_list), len(result_list[0].prior_comp_time)])
        
        i = 0
        obj_keys = []
        for r in result_list:
            ua_reward[i,:] = r.ua_reward
            ts_reward[i,:] = r.ts_reward
            gi_reward[i,:] = r.gi_reward
            ts_corr_reward[i,:] = r.ts_corr_reward
            bucb_corr_reward[i,:] = r.bucb_corr_reward
            for n, ts_corr_prior_reward in enumerate(ts_corr_prior_rewards):
                ts_corr_prior_reward[i,:] = r.ts_corr_prior_reward[n]
            for n, bucb_corr_prior_reward in enumerate(bucb_corr_prior_rewards):
                bucb_corr_prior_reward[i,:] = r.bucb_corr_prior_reward[n]

            ce_vals[i,:] = r.ce_vals
            se_vals[i,:] = r.se_vals
            we_vals[i,:] = r.we_vals
            num_grasps[i] = r.num_grasps
            total_weights[i] = r.total_weights


            ua_runtime[i] = r.ua_runtime
            ts_runtime[i] = r.ts_runtime
            ts_corr_runtime[i] = r.ts_corr_runtime
            bucb_corr_runtime[i] = r.bucb_corr_runtime
            ts_corr_prior_runtime[i,:] = np.asarray(r.ts_corr_prior_runtime)
            bucb_corr_prior_runtime[i,:] = np.asarray(r.bucb_corr_prior_runtime)
            prior_comp_time[i,:] = np.asarray(r.prior_comp_time)

            obj_keys.append(r.obj_key)
            i = i + 1

        true_avg_rewards = [r.true_avg_reward for r in result_list]
        iters = [r.iters for r in result_list]
        kernel_matrices = [r.kernel_matrix for r in result_list]

        neighbor_kernels = [r.neighbor_kernels for r in result_list]
        neighbor_pfc_diffs = [r.neighbor_pfc_diffs for r in result_list]
        neighbor_distances = [r.neighbor_distances for r in result_list]
        neighbor_keys = [r.neighbor_keys for r in result_list]

        return BanditCorrelatedPriorExperimentResult(ua_reward, ts_reward, gi_reward, ts_corr_reward, bucb_corr_reward, ts_corr_prior_rewards, bucb_corr_prior_rewards,
                                                     true_avg_rewards, iters, kernel_matrices, neighbor_kernels, neighbor_pfc_diffs,
                                                     neighbor_keys, ce_vals, se_vals, we_vals, num_grasps, total_weights,
                                                     [], [], [], [], [], [],
                                                     ua_runtime, ts_runtime, ts_corr_runtime, bucb_corr_runtime, ts_corr_prior_runtime, bucb_corr_prior_runtime,
                                                     prior_comp_time,
                                                     obj_keys, neighbor_keys, len(result_list))

def load_candidate_grasps(obj, chunk):
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

    # run bandits!
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    candidates = []
    out_rate = 50
    for k, (grasp, features) in enumerate(zip(grasps, all_features)):
        if k % out_rate == 0:
            logging.info('Adding grasp %d' %(k))
        grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
        pfc_rv = pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config)
        if features is None:
            logging.info('Could not compute features for grasp.')
        else:
            pfc_rv.set_features(features)
            candidates.append(pfc_rv)

    return candidates


def label_correlated(obj, chunk, config, plot=False,
                     priors_dataset=None, nearest_features_names=None):
    """Label an object with grasps according to probability of force closure,
    using correlated bandits."""
    # bandit params
    num_trials = config['num_trials']
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]

    bandit_start = time.clock()

    np.random.seed(100)

    candidates = load_candidate_grasps(obj, chunk)
    if candidates is None:
        return None

    # feature transform
    def phi(rv):
        return rv.features

    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
    objective = objectives.RandomBinaryObjective()

    # compute priors
    logging.info('Computing priors')
    if priors_dataset is None:
        priors_dataset = chunk
    prior_engine = pce.PriorComputationEngine(priors_dataset, config)

    # Compute priors
    all_alpha_priors = []
    all_beta_priors = []
    prior_comp_times = []
    if nearest_features_names == None:
        alpha_priors, beta_priors = prior_engine.compute_priors(obj, candidates)
        all_alpha_priors.append(alpha_priors)
        all_beta_priors.append(beta_priors)
    else:
        for nearest_features_name in nearest_features_names:
            logging.info('Computing priors using %s' %(nearest_features_name))
            priors_start_time = time.time()
            alpha_priors, beta_priors, neighbor_keys, neighbor_distances, neighbor_kernels, neighbor_pfc_diffs, num_grasp_neighbors = \
                prior_engine.compute_priors(obj, candidates, nearest_features_name=nearest_features_name)

            all_alpha_priors.append(alpha_priors)
            all_beta_priors.append(beta_priors)
            priors_end_time = time.time()
            prior_comp_times.append(priors_end_time - priors_start_time)
            logging.info('Priors for %s took %f' %(nearest_features_name, priors_end_time - priors_start_time))

    # pre-computed pfc values
    logging.info('Computing regression errors')
    true_pfc = np.array([c.grasp.quality for c in candidates])
    prior_alphas = np.ones(true_pfc.shape)
    prior_betas = np.ones(true_pfc.shape)
    prior_pfc = 0.5*np.ones(true_pfc.shape)

    ce_loss = objectives.CrossEntropyLoss(true_pfc)
    se_loss = objectives.SquaredErrorLoss(true_pfc)
    we_loss = objectives.WeightedSquaredErrorLoss(true_pfc)
    ccbp_ll = objectives.CCBPLogLikelihood(true_pfc)
    ce_vals = [ce_loss(prior_pfc)]
    se_vals = [se_loss(prior_pfc)]
    we_vals = [se_loss(prior_pfc)] # uniform weights at first
    ccbp_vals = [ccbp_ll.evaluate(prior_alphas, prior_betas)]
    total_weights = [len(candidates)]

    # compute estimated pfc values from alphas and betas
    for alpha_prior, beta_prior in zip(all_alpha_priors, all_beta_priors):
        estimated_pfc = models.BetaBernoulliModel.beta_mean(np.array(alpha_prior), np.array(beta_prior))
        estimated_vars = models.BetaBernoulliModel.beta_variance(np.array(alpha_prior), np.array(beta_prior))
 
        # compute losses
        ce_vals.append(ce_loss(estimated_pfc))
        se_vals.append(se_loss(estimated_pfc))
        we_vals.append(we_loss.evaluate(estimated_pfc, estimated_vars))
        ccbp_vals.append(ccbp_ll.evaluate(np.array(alpha_prior), np.array(beta_prior)))
        total_weights.append(np.sum(estimated_vars))

    ce_vals = np.array(ce_vals)
    se_vals = np.array(se_vals)
    we_vals = np.array(we_vals)
    ccbp_vals = np.array(ccbp_vals)
    total_weights = np.array(total_weights)

    # setup reward buffers
    ua_rewards = []
    ts_rewards = []
    gi_rewards = []
    ts_corr_rewards = []
    bucb_corr_rewards = []
    all_ts_corr_prior_rewards = []
    for x in range(0, len(all_alpha_priors)):
        all_ts_corr_prior_rewards.append([])
    all_bucb_corr_prior_rewards = []
    for x in range(0, len(all_alpha_priors)):
        all_bucb_corr_prior_rewards.append([])

    # setup runtime buffers
    ua_runtimes = []
    ts_runtimes = []
    gi_runtimes = []
    ts_corr_runtimes = []
    bucb_corr_runtimes = []
    all_ts_corr_prior_runtimes = []
    for x in range(0, len(all_alpha_priors)):
        all_ts_corr_prior_runtimes.append([])
    all_bucb_corr_prior_runtimes = []
    for x in range(0, len(all_alpha_priors)):
        all_bucb_corr_prior_runtimes.append([])

    # run bandits for several trials
    logging.info('Running bandits')
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

        # Gittins indices
        gi = das.GittinsIndex98(objective, candidates)
        logging.info('Running Gittins Indices.')
        gi_result = gi.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # correlated Thompson sampling for even faster convergence
        ts_corr = das.CorrelatedThompsonSampling(
            objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'], p=config['lb_alpha'])
        logging.info('Running correlated Thompson sampling.')
        ts_corr_result = ts_corr.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # correlated Thompson sampling for even faster convergence
        bucb_corr = das.CorrelatedGittins(
            objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'], p=config['lb_alpha'])#horizon=max_iter)
        logging.info('Running correlated Bayes UCB.')
        bucb_corr_result = bucb_corr.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

        # correlated MAB for faster convergence
        all_ts_corr_prior_ind = []
        all_bucb_corr_prior_ind = []
        for alpha_priors, beta_priors, ts_corr_prior_rewards, bucb_corr_prior_rewards, ts_corr_runtimes, bucb_corr_runtimes, nearest_features_name in \
                zip(all_alpha_priors, all_beta_priors, all_ts_corr_prior_rewards, all_bucb_corr_prior_rewards, all_ts_corr_prior_runtimes, all_bucb_corr_prior_runtimes, nearest_features_names):
            # thompson sampling
            ts_corr_prior = das.CorrelatedThompsonSampling(
                objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'], alpha_prior = alpha_priors,
                beta_prior = beta_priors, p=config['lb_alpha'])
            logging.info('Running correlated Thompson sampling with priors from %s' %(nearest_features_name))
            ts_corr_prior_result = ts_corr_prior.solve(termination_condition=tc.OrTerminationCondition(tc_list),
                                                       snapshot_rate=snapshot_rate)
            ts_corr_prior_normalized_reward = reward_vs_iters(ts_corr_prior_result, true_pfc)
            ts_corr_prior_rewards.append(ts_corr_prior_normalized_reward)
            ts_corr_runtimes.append(ts_corr_prior_result.total_time)
            all_ts_corr_prior_ind.append(ts_corr_prior_result.best_pred_ind)

            # bayes ucb
            bucb_corr = das.CorrelatedGittins(
                objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'], #horizon=max_iter,
                alpha_prior = alpha_priors, beta_prior = beta_priors, p=config['lb_alpha'])
            logging.info('Running correlated Bayes UCB with priors from %s' %(nearest_features_name))
            bucb_corr_prior_result = bucb_corr.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)
            bucb_corr_prior_normalized_reward = reward_vs_iters(bucb_corr_prior_result, true_pfc)
            bucb_corr_prior_rewards.append(bucb_corr_prior_normalized_reward)
            bucb_corr_runtimes.append(bucb_corr_prior_result.total_time)
            all_bucb_corr_prior_ind.append(bucb_corr_prior_result.best_pred_ind)

        # compile results
        ua_normalized_reward = reward_vs_iters(ua_result, true_pfc)
        ts_normalized_reward = reward_vs_iters(ts_result, true_pfc)
        gi_normalized_reward = reward_vs_iters(gi_result, true_pfc)
        ts_corr_normalized_reward = reward_vs_iters(ts_corr_result, true_pfc)
        bucb_corr_normalized_reward = reward_vs_iters(bucb_corr_result, true_pfc)

        ua_rewards.append(ua_normalized_reward)
        ts_rewards.append(ts_normalized_reward)
        gi_rewards.append(gi_normalized_reward)
        ts_corr_rewards.append(ts_corr_normalized_reward)
        bucb_corr_rewards.append(bucb_corr_normalized_reward)

        ua_runtimes.append(ua_result.total_time)
        ts_runtimes.append(ts_result.total_time)
        gi_runtimes.append(gi_result.total_time)
        ts_corr_runtimes.append(ts_corr_result.total_time)
        bucb_corr_runtimes.append(bucb_corr_result.total_time)

    if num_trials == 0:
        return None

    # get the bandit rewards
    all_ua_rewards = np.array(ua_rewards)
    all_ts_rewards = np.array(ts_rewards)
    all_gi_rewards = np.array(gi_rewards)
    all_ts_corr_rewards = np.array(ts_corr_rewards)
    all_bucb_corr_rewards = np.array(bucb_corr_rewards)

    all_avg_ts_corr_prior_rewards = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        all_avg_ts_corr_prior_rewards.append(np.mean(np.array(ts_corr_prior_rewards), axis=0))

    all_avg_bucb_corr_prior_rewards = []
    for bucb_corr_prior_rewards in all_bucb_corr_prior_rewards:
        all_avg_bucb_corr_prior_rewards.append(np.mean(np.array(bucb_corr_prior_rewards), axis=0))
        #all_avg_bucb_corr_prior_rewards.append([])

    # get bandit indices
    ua_ind = ua_result.best_pred_ind
    ts_ind = ts_result.best_pred_ind
    ts_corr_ind = ts_corr_result.best_pred_ind
    bucb_corr_ind = bucb_corr_result.best_pred_ind

    # compute avg normalized rewards
    avg_ua_rewards = np.mean(all_ua_rewards, axis=0)
    avg_ts_rewards = np.mean(all_ts_rewards, axis=0)
    avg_gi_rewards = np.mean(all_gi_rewards, axis=0)
    avg_ts_corr_rewards = np.mean(all_ts_corr_rewards, axis=0)
    avg_bucb_corr_rewards = np.mean(all_bucb_corr_rewards, axis=0)
    #avg_bucb_corr_rewards = all_bucb_corr_rewards

    # compute avg runtimes
    avg_ua_runtimes = np.mean(np.array(ua_runtimes), axis=0)
    avg_ts_runtimes = np.mean(np.array(ts_runtimes), axis=0)
    avg_ts_corr_runtimes = np.mean(np.array(ts_corr_runtimes), axis=0)
    avg_bucb_corr_runtimes = np.mean(np.array(bucb_corr_runtimes), axis=0)
    all_avg_ts_corr_prior_runtimes = []
    for ts_corr_prior_runtimes in all_ts_corr_prior_runtimes:
        all_avg_ts_corr_prior_runtimes.append(np.mean(np.array(ts_corr_prior_runtimes), axis=0))
    all_avg_bucb_corr_prior_runtimes = []    
    for bucb_corr_prior_runtimes in all_bucb_corr_prior_runtimes:
        all_avg_bucb_corr_prior_runtimes.append(np.mean(np.array(bucb_corr_prior_runtimes), axis=0))

    # kernel matrix
    kernel_matrix = kernel.matrix(candidates)

    return BanditCorrelatedPriorExperimentResult(avg_ua_rewards, avg_ts_rewards, avg_gi_rewards, avg_ts_corr_rewards, avg_bucb_corr_rewards,
                                                 all_avg_ts_corr_prior_rewards, all_avg_bucb_corr_prior_rewards,
                                                 true_pfc, ua_result.iters, kernel_matrix,
                                                 [], [], [],
                                                 ce_vals, ccbp_vals, we_vals, len(candidates), total_weights,
                                                 ua_ind, ts_ind, ts_corr_ind, bucb_corr_ind, all_ts_corr_prior_ind, all_bucb_corr_prior_ind,
                                                 avg_ua_runtimes, avg_ts_runtimes, avg_ts_corr_runtimes, avg_bucb_corr_runtimes, all_avg_ts_corr_prior_runtimes, all_avg_bucb_corr_prior_runtimes,
                                                 prior_comp_times, obj_key=obj.key, neighbor_keys=neighbor_keys)

def plot_kernels_for_key(obj, chunk, config, priors_dataset=None, nearest_features_name=None):
    candidates = load_candidate_grasps(obj, chunk)

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
    candidates = load_candidate_grasps(obj, chunk)

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
    nearest_features_names = config['priors_feature_names']

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
    all_results = BanditCorrelatedPriorExperimentResult.compile_results(results)

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
    for ts_corr_prior, label in zip(ts_corr_prior_normalized_reward, nearest_features_names):
        plt.plot(all_results.iters[0], ts_corr_prior,
                 c=u'c', linewidth=2.0, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))
    plt.xlim(0, np.max(all_results.iters[0]))
    plt.ylim(0.5, 1)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(result_dir, obj.key+'_results.png'), dpi=dpi, bbox_extra_artists=(legend,), bbox_inches='tight')

    # plot kernels
    for nearest_features_name in nearest_features_names:
        plot_kernels_for_key(obj, chunk, config, priors_dataset, nearest_features_names[0])
        fname = nearest_features_name.replace('nearest_features', '%s_kernels' %(obj.key))
        plt.savefig(os.path.join(result_dir, fname), dpi=dpi)

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

    # make output directory
    dest = os.path.join(args.output_dest, chunk.name)
    try:
        os.makedirs(dest)
    except os.error:
        pass

    if 'priors_dataset' in config:
        priors_dataset = db.Dataset(config['priors_dataset'], config)
    else:
        priors_dataset = None

    # loop through objects, labelling each
    results = []
    for obj in chunk:
        if obj.key in skip_keys:
            continue

        logging.info('Labelling object {}'.format(obj.key))
        experiment_result = label_correlated(obj, chunk, config,
                                             priors_dataset=priors_dataset,
                                             nearest_features_names=config['priors_feature_names'])
        if experiment_result is None:
            continue
        results.append(experiment_result)

    if len(results) == 0:
        logging.info('Exiting. No grasps found')
        exit(0)

    # save to file
    logging.info('Saving results to %s' %(dest))
    for r in results:
        r.save(dest)

    if config['plot']:
        # combine results
        all_results = BanditCorrelatedPriorExperimentResult.compile_results(results)
