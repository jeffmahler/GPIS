import argparse
import IPython
import logging
import matplotlib as mpl; mpl.use('Agg') # this doesn't seem to work...
import matplotlib.pyplot as plt
import models
import numpy as np
import pickle as pkl
import os
import sys
import scipy.spatial.distance as ssd

import correlated_bandits_priors as cb
from correlated_bandits_priors import BanditCorrelatedExperimentResult
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

    # aggregate results
    if len(results) == 0:
        exit(0)

    all_results = BanditCorrelatedExperimentResult.compile_results(results)

    # plot params
    line_width = config['line_width']
    font_size = config['font_size']
    dpi = config['dpi']

    # plotting of final results
    ua_normalized_reward = np.mean(all_results.ua_reward, axis=0)
    ts_normalized_reward = np.mean(all_results.ts_reward, axis=0)
    ts_corr_normalized_reward = np.mean(all_results.ts_corr_reward, axis=0)

    all_ts_corr_prior_rewards = all_results.ts_corr_prior_reward
    ts_corr_prior_normalized_reward = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        ts_corr_prior_normalized_reward.append(np.mean(ts_corr_prior_rewards, axis=0))

    plt.figure()
    plt.plot(all_results.iters[0], ua_normalized_reward, c=u'b', linewidth=2.0, label='Uniform')
    plt.plot(all_results.iters[0], ts_normalized_reward, c=u'g', linewidth=2.0, label='TS (Uncorrelated)')
    plt.plot(all_results.iters[0], ts_corr_normalized_reward, c=u'r', linewidth=2.0, label='TS (Correlated)')

    for ts_corr_prior, color, label in zip(ts_corr_prior_normalized_reward, u'cmb',
                                           config['priors_feature_names']):
        plt.plot(all_results.iters[0], ts_corr_prior,
                 c=color, linewidth=2.0, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))

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

    exit(0)

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
