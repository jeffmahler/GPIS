import IPython
import logging
import matplotlib.pyplot as plt
import models
import numpy as np
import pickle as pkl
import os
import sys

import correlated_bandits as cb
from correlated_bandits import BanditCorrelatedExperimentResult
import experiment_config as ec

if __name__ == '__main__':
    config_file = sys.argv[1]
    result_dir = sys.argv[2]

    logging.getLogger().setLevel(logging.INFO)
    config = ec.ExperimentConfig(config_file)

    # read in all pickle files
    results = []
    for _, dirs, _ in os.walk(result_dir):
        # compile each subdirectory
        for d in dirs:
            # get the pickle files from each directory
            for root, _, files in os.walk(os.path.join(result_dir, d)):
                for f in files:
                    if f.endswith('.pkl'):
                        result_pkl = os.path.join(root, f)
                        f = open(result_pkl, 'r')
                        
                        logging.info('Reading %s' %(result_pkl))
                        p = pkl.load(f)
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

    # plot the prior distribution of grasp quality
    grasp_qualities = np.zeros(0)
    for result in results:
        final_model = result.ua_result.models[-1]
        pfc = models.BetaBernoulliModel.beta_mean(final_model.alphas, final_model.betas)
        grasp_qualities = np.r_[grasp_qualities, pfc]

    # plot histograms
    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins+1)
    plt.figure()
    n, bins, patches = plt.hist(grasp_qualities, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)

    figname = 'histogram_success.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)

    # plotting of final results
    ua_avg_norm_reward = np.mean(all_results.ua_reward, axis=0)
    ts_avg_norm_reward = np.mean(all_results.ts_reward, axis=0)
    ts_corr_avg_norm_reward = np.mean(all_results.ts_corr_reward, axis=0)

    ua_std_norm_reward = np.std(all_results.ua_reward, axis=0)
    ts_std_norm_reward = np.std(all_results.ts_reward, axis=0)
    ts_corr_std_norm_reward = np.std(all_results.ts_corr_reward, axis=0)

    # plot avg simple regret
    plt.figure()

    plt.plot(all_results.ua_result[0].iters, ua_avg_norm_reward, c=u'b', linewidth=line_width, label='Uniform Allocation')
    plt.plot(all_results.ts_result[0].iters, ts_avg_norm_reward, c=u'g', linewidth=line_width, label='Thompson Sampling (Uncorrelated)')
    plt.plot(all_results.ts_corr_result[0].iters, ts_corr_avg_norm_reward, c=u'r', linewidth=line_width, label='Thompson Sampling (Correlated)')

    plt.xlim(0, np.max(all_results.ts_result[0].iters))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    figname = 'avg_reward.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)

    # plot avg simple regret w error bars
    plt.figure()

    plt.errorbar(all_results.ua_result[0].iters, ua_avg_norm_reward, yerr=ua_std_norm_reward, c=u'b', linewidth=line_width, label='Uniform Allocation')
    plt.errorbar(all_results.ts_result[0].iters, ts_avg_norm_reward, yerr=ts_std_norm_reward, c=u'g', linewidth=line_width, label='Thompson Sampling (Uncorrelated)')
    plt.errorbar(all_results.ts_corr_result[0].iters, ts_corr_avg_norm_reward, yerr=ts_corr_std_norm_reward, c=u'r', linewidth=line_width, label='Thompson Sampling (Correlated)')

    plt.xlim(0, np.max(all_results.ts_result[0].iters))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC with StdDev vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    figname = 'avg_reward_with_error_bars.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)

    # finally, show
    plt.show()
