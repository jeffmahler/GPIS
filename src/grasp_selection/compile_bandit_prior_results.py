"""
Generates plots for the prior correlated bandits experiments
"""
import argparse
import IPython
import logging
import matplotlib as mpl; mpl.use('Agg') # this doesn't seem to work...
import matplotlib.pyplot as plt
import models
import numpy as np
import plotting
import pickle as pkl
import os
import sys
import scipy.spatial.distance as ssd

import correlated_bandits_priors as cb
from correlated_bandits_priors import BanditCorrelatedPriorExperimentResult
import experiment_config as ec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('result_dir')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    config = ec.ExperimentConfig(args.config)
    result_dir = args.result_dir

    # read in all pickle files
    results = []
    names = []
    for _, dirs, _ in os.walk(result_dir):
        # compile each subdirectory
        for d in dirs:
            # get the pickle files from each directory
            for root, _, files in os.walk(os.path.join(result_dir, d)):
                for f in files:
                    if f.endswith('.pkl'):
                        names.append(f.split('.')[0])
                        result_pkl = os.path.join(root, f)
                        f = open(result_pkl, 'r')

                        logging.info('Reading %s' %(result_pkl))
                        try:
                            p = pkl.load(f)
                        except:
                            continue

                        if p is not None:
                            results.append(p)

    if len(results) == 0:
        exit(0)

    # plot params
    line_width = config['line_width']
    font_size = config['font_size']
    dpi = config['dpi']

    # per-object plots
    for result in results:
        # kernel plot
        pfc_arr = np.array([result.true_avg_reward]).T
        pfc_diff = ssd.squareform(ssd.pdist(pfc_arr))
        plotting.plot_kernels(result.obj_key, result.kernel_matrix, pfc_diff,
                              result.neighbor_kernels, result.neighbor_pfc_diffs, result.neighbor_keys,
                              font_size=font_size)
        figname = '%s_kernels.png' %(result.obj_key)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        # grasp histogram plot
        plotting.plot_grasp_histogram(result.true_avg_reward, font_size=font_size)
        figname = '%s_grasp_histogram.png' %(result.obj_key)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)
        
        # avg reward plot
        plt.figure()
        plt.plot(result.iters, result.ua_reward, c=u'b', linewidth=line_width, label='Uniform Allocation')
        plt.plot(result.iters, result.ts_reward, c=u'g', linewidth=line_width, label='Thompson Sampling (Uncorrelated)')
        plt.plot(result.iters, result.ts_corr_reward, c=u'r', linewidth=line_width, label='Thompson Sampling (Correlated)')
        for ts_corr_prior, color, label in zip(result.ts_corr_prior_reward, u'cmb',
                                               config['priors_feature_names']):
            plt.plot(result.iters, ts_corr_prior,
                     c=color, linewidth=line_width, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right')

        figname = '%s_avg_reward.png' %(result.obj_key)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

    # aggregate results
    all_results = BanditCorrelatedPriorExperimentResult.compile_results(results)

    # plotting of average final results
    ua_normalized_reward = np.mean(all_results.ua_reward, axis=0)
    ts_normalized_reward = np.mean(all_results.ts_reward, axis=0)
    ts_corr_normalized_reward = np.mean(all_results.ts_corr_reward, axis=0)

    all_ts_corr_prior_rewards = all_results.ts_corr_prior_reward
    ts_corr_prior_normalized_reward = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        ts_corr_prior_normalized_reward.append(np.mean(ts_corr_prior_rewards, axis=0))

    plt.figure()
    plt.plot(all_results.iters[0], ua_normalized_reward, c=u'b', linewidth=line_width, label='Uniform')
    plt.plot(all_results.iters[0], ts_normalized_reward, c=u'g', linewidth=line_width, label='TS (Uncorrelated)')
    plt.plot(all_results.iters[0], ts_corr_normalized_reward, c=u'r', linewidth=line_width, label='TS (Correlated)')

    for ts_corr_prior, color, label in zip(ts_corr_prior_normalized_reward, u'cmb',
                                           config['priors_feature_names']):
        plt.plot(all_results.iters[0], ts_corr_prior,
                 c=color, linewidth=line_width, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))

    plt.xlim(0, np.max(all_results.iters[0]))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    figname = 'avg_reward.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)
