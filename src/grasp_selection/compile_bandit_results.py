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

    correlations_dir = os.path.join(result_dir, 'correlations')
    if not os.path.exists(correlations_dir):
        os.mkdir(correlations_dir)

    # plot the prior distribution of grasp quality
    grasp_qualities = np.zeros(0)
    grasp_qualities_diff = np.zeros(0)
    kernel_values = np.zeros(0)
    for obj_name, result in zip(names, results):
        ua_final_model = result.ua_result.models[-1]
        pfc = models.BetaBernoulliModel.beta_mean(ua_final_model.alphas, ua_final_model.betas)
        grasp_qualities = np.r_[grasp_qualities, pfc]

        ts_corr_final_model = result.ts_corr_result.models[-1]
        pfc_diff_vec = ssd.squareform(ssd.pdist(np.array([pfc]).T)).ravel()
        k_vec = ts_corr_final_model.correlations.ravel()
        plt.figure()
        plt.scatter(k_vec, pfc_diff_vec)
        plt.xlabel('Kernel', fontsize=font_size)
        plt.ylabel('PFC Diff', fontsize=font_size)
        plt.title('%s Correlations' %(obj_name), fontsize=font_size)
        figname = 'correlations_%s.png' %(obj_name)
        plt.savefig(os.path.join(correlations_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        grasp_qualities_diff = np.r_[grasp_qualities_diff, pfc_diff_vec]
        kernel_values = np.r_[kernel_values, k_vec]

    # plot pfc difference
    # plt.figure()
    # plt.scatter(kernel_values, grasp_qualities_diff)
    # plt.xlabel('Kernel', fontsize=font_size)
    # plt.ylabel('PFC Diff', fontsize=font_size)
    # plt.title('Correlations', fontsize=font_size)

    # figname = 'correlations.png'
    # plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    # logging.info('Finished plotting %s', figname)

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
    logging.info('Finished plotting %s', figname)

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
    logging.info('Finished plotting %s', figname)

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
    logging.info('Finished plotting %s', figname)

    # finally, show
    if config['plot']:
        plt.show()
