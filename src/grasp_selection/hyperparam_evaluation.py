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

class HyperparamEvalResult:
    def __init__(self, ce_vals, se_vals, we_vals, ccbp_ll_vals, num_grasps, total_weights, hyperparams,
                 prior_comp_time, obj_key='', neighbor_keys=[], num_objects=1):
        self.ce_vals = ce_vals
        self.se_vals = se_vals
        self.we_vals = we_vals
        self.ccbp_ll_vals = ccbp_ll_vals
        self.num_grasps = num_grasps
        self.total_weights = total_weights

        self.hyperparams = hyperparams
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

        ce_vals = np.zeros([len(result_list), result_list[0].ce_vals.shape[0]])
        se_vals = np.zeros([len(result_list), result_list[0].se_vals.shape[0]])
        we_vals = np.zeros([len(result_list), result_list[0].we_vals.shape[0]])
        ccbp_ll_vals = np.zeros([len(result_list), result_list[0].ccbp_ll_vals.shape[0]])
        num_grasps = np.zeros(len(result_list))
        total_weights = np.zeros([len(result_list), result_list[0].total_weights.shape[0]])

        prior_comp_time = np.zeros([len(result_list), len(result_list[0].prior_comp_time)])
        
        i = 0
        obj_keys = []
        for r in result_list:
            ce_vals[i,:] = r.ce_vals
            se_vals[i,:] = r.se_vals
            we_vals[i,:] = r.we_vals
            ccbp_ll_vals[i,:] = r.ccbp_ll_vals
            num_grasps[i] = r.num_grasps
            total_weights[i] = r.total_weights

            prior_comp_time[i,:] = np.asarray(r.prior_comp_time)

            obj_keys.append(r.obj_key)
            i = i + 1

        hyperparams = [r.hyperparams for r in result_list]
        neighbor_keys = [r.neighbor_keys for r in result_list]

        return HyperparamEvalResult(ce_vals, se_vals, we_vals, ccbp_ll_vals, num_grasps, total_weights, hyperparams,
                                    prior_comp_time, obj_keys, neighbor_keys, len(result_list))

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


def eval_hyperparams(obj, chunk, config, plot=False,
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

    # create hyperparam dict
    num_grasps = len(candidates)
    hyperparams = {}
    hyperparams['weight_grad'] = config['weight_grad_x']
    hyperparams['weight_moment'] = config['weight_gravity']
    hyperparams['weight_shape'] = config['prior_neighbor_weight']
    hyperparams['num_neighbors'] = config['prior_num_neighbors']
    return HyperparamEvalResult(ce_vals, se_vals, we_vals, ccbp_vals, num_grasps, total_weights, hyperparams,
                                prior_comp_times, obj_key=obj.key, neighbor_keys=neighbor_keys)

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
        experiment_result = eval_hyperparams(obj, chunk, config,
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
