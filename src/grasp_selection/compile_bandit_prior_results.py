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

ignore_keys = []#'BigBIRD_dove_go_fresh_burst', 'KIT_Shampoo_800_tex', 'KIT_Wafflerolls_800_tex', 'ModelNet40_radio_102'] # for data cleaning

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

                        if p is not None and p.obj_key not in ignore_keys:
                            results.append(p)

    if len(results) == 0:
        exit(0)

    # plot params
    line_width = 1.0#config['line_width']
    font_size = config['font_size']
    dpi = config['dpi']

    num_colors = 5 + len(results[0].ts_corr_prior_reward) + len(results[0].bucb_corr_prior_reward)
    colors = plotting.distinguishable_colors(num_colors)

    # per-object plots
    """
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
        plt.plot(result.iters, result.ua_reward, c=colors[0], linewidth=line_width, label='Uniform Allocation')
        plt.plot(result.iters, result.ts_reward, c=colors[1], linewidth=line_width, label='Thompson Sampling (Uncorrelated)')
        plt.plot(result.iters, result.gi_reward, c=colors[2], linewidth=line_width, label='Gittins Indices')
        plt.plot(result.iters, result.ts_corr_reward, c=colors[3], linewidth=line_width, label='Thompson Sampling (Correlated)')
        plt.plot(result.iters, result.bucb_corr_reward, c=colors[4], linewidth=line_width, label='Bayes UCB (Correlated)')
        for ts_corr_prior, color, label in zip(result.ts_corr_prior_reward, colors[5:5+len(result.ts_corr_prior_reward)],
                                               config['priors_feature_names']):
            plt.plot(result.iters, ts_corr_prior,
                     c=color, linewidth=line_width, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))

        for bucb_corr_prior, color, label in zip(result.bucb_corr_prior_reward, colors[5+len(result.bucb_corr_prior_reward):],
                                                 config['priors_feature_names']):
            plt.plot(result.iters, bucb_corr_prior,
                     c=color, linewidth=line_width, label='BUCB (%s)' %(label.replace('nearest_features', 'Priors')))

        plt.xlim(0, np.max(result.iters))
        plt.ylim(0.5, 1)
        plt.xlabel('Iteration', fontsize=font_size)
        plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
        plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right', prop={'size':5})

        figname = '%s_avg_reward.pdf' %(result.obj_key)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

    """

    # aggregate results
    all_results = BanditCorrelatedPriorExperimentResult.compile_results(results)

    # plot ce and se
    avg_ce = np.sum(np.diag(all_results.num_grasps).dot(all_results.ce_vals), axis=0) * (1.0 / np.sum(all_results.num_grasps))
    avg_se = np.sum(np.diag(all_results.num_grasps).dot(all_results.se_vals), axis=0) * (1.0 / np.sum(all_results.num_grasps))
    avg_we = np.sum(all_results.total_weights * all_results.we_vals, axis=0)  / np.sum(all_results.total_weights, axis=0)
    np.savetxt('cross_entropy_vs_nn.csv', avg_ce, delimiter=',')
    np.savetxt('squared_error_vs_nn.csv', avg_se, delimiter=',')
    np.savetxt('weighted_squared_error_vs_nn.csv', avg_we, delimiter=',')

    plt.figure()
    plt.plot(avg_ce, c=u'b', linewidth=line_width)
    plt.xlabel('Prior Data', fontsize=font_size)
    plt.ylabel('Cross Entropy', fontsize=font_size)
    plt.title('Cross Entropy vs Prior Data', fontsize=font_size)
    figname = 'cross_entropy_vs_nn.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)

    plt.figure()
    plt.plot(avg_se, c=u'b', linewidth=line_width)
    plt.xlabel('Prior Data', fontsize=font_size)
    plt.ylabel('Squared Error', fontsize=font_size)
    plt.title('Squared Error vs Prior Data', fontsize=font_size)
    figname = 'squared_error_vs_nn.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)

    plt.figure()
    plt.plot(avg_we, c=u'b', linewidth=line_width)
    plt.xlabel('Prior Data', fontsize=font_size)
    plt.ylabel('Weighted Squared Error', fontsize=font_size)
    plt.title('Weighted Squared Error vs Prior Data', fontsize=font_size)
    figname = 'weighted_squared_error_vs_nn.png'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)

    # plotting of average final results
    """
    ua_normalized_reward = np.median(all_results.ua_reward, axis=0)
    ts_normalized_reward = np.median(all_results.ts_reward, axis=0)
    gi_normalized_reward = np.median(all_results.gi_reward, axis=0)
    ts_corr_normalized_reward = np.median(all_results.ts_corr_reward, axis=0)
    bucb_corr_normalized_reward = np.median(all_results.bucb_corr_reward, axis=0)

    all_ts_corr_prior_rewards = all_results.ts_corr_prior_reward
    ts_corr_prior_normalized_reward = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        ts_corr_prior_normalized_reward.append(np.median(ts_corr_prior_rewards, axis=0))

    all_bucb_corr_prior_rewards = all_results.bucb_corr_prior_reward
    bucb_corr_prior_normalized_reward = []
    for bucb_corr_prior_rewards in all_bucb_corr_prior_rewards:
        bucb_corr_prior_normalized_reward.append(np.median(bucb_corr_prior_rewards, axis=0))
    """

    # avg normalized reward
    ua_normalized_reward = np.percentile(all_results.ua_reward, 50, axis=0)
    ts_normalized_reward = np.percentile(all_results.ts_reward, 50, axis=0)
    gi_normalized_reward = np.percentile(all_results.gi_reward, 50, axis=0)
    ts_corr_normalized_reward = np.percentile(all_results.ts_corr_reward, 50, axis=0)
    bucb_corr_normalized_reward = np.percentile(all_results.bucb_corr_reward, 50, axis=0)

    all_ts_corr_prior_rewards = all_results.ts_corr_prior_reward
    ts_corr_prior_normalized_reward = []
    for ts_corr_prior_rewards in all_ts_corr_prior_rewards:
        ts_corr_prior_normalized_reward.append(np.percentile(ts_corr_prior_rewards, 50, axis=0))

    all_bucb_corr_prior_rewards = all_results.bucb_corr_prior_reward
    bucb_corr_prior_normalized_reward = []
    for bucb_corr_prior_rewards in all_bucb_corr_prior_rewards:
        bucb_corr_prior_normalized_reward.append(np.percentile(bucb_corr_prior_rewards, 50, axis=0))

    # get standard error on the mean
    ua_normalized_std_reward = np.percentile(all_results.ua_std_reward, 50, axis=0)
    ts_normalized_std_reward = np.percentile(all_results.ts_std_reward, 50, axis=0)
    gi_normalized_std_reward = np.percentile(all_results.gi_std_reward, 50, axis=0)
    ts_corr_normalized_std_reward = np.percentile(all_results.ts_corr_std_reward, 50, axis=0)
    bucb_corr_normalized_std_reward = np.percentile(all_results.bucb_corr_std_reward, 50, axis=0)

    all_ts_corr_prior_std_rewards = all_results.ts_corr_prior_std_reward
    ts_corr_prior_normalized_std_reward = []
    for ts_corr_prior_std_rewards in all_ts_corr_prior_std_rewards:
        ts_corr_prior_normalized_std_reward.append(np.percentile(ts_corr_prior_std_rewards, 50, axis=0))

    all_bucb_corr_prior_std_rewards = all_results.bucb_corr_prior_std_reward
    bucb_corr_prior_normalized_std_reward = []
    for bucb_corr_prior_std_rewards in all_bucb_corr_prior_std_rewards:
        bucb_corr_prior_normalized_std_reward.append(np.percentile(bucb_corr_prior_std_rewards, 50, axis=0))

    num_objects = ua_normalized_reward.shape[0]
    num_trials = config['num_trials']
    n = num_objects * num_trials
    z = 1.0 / np.sqrt(n)

    ua_normalized_std_reward = z * ua_normalized_std_reward
    ts_normalized_std_reward = z * ts_normalized_std_reward
    gi_normalized_std_reward = z * gi_normalized_std_reward
    ts_corr_normalized_std_reward = z * ts_corr_normalized_std_reward
    bucb_corr_normalized_std_reward = z * bucb_corr_normalized_std_reward
    ts_corr_prior_normalized_std_reward = [z * t for t in ts_corr_prior_normalized_std_reward]
    bucb_corr_prior_normalized_std_reward = [z * t for t in bucb_corr_prior_normalized_std_reward]

    # plot normalized rewards
    plt.figure()
    plt.plot(all_results.iters[0], ua_normalized_reward, c=colors[0], linewidth=line_width, label='Uniform')
    plt.plot(all_results.iters[0], ts_normalized_reward, c=colors[1], linewidth=line_width, label='TS (Uncorrelated)')
    plt.plot(all_results.iters[0], gi_normalized_reward, c=colors[2], linewidth=line_width, label='Gittins Indices')
    plt.plot(all_results.iters[0], ts_corr_normalized_reward, c=colors[3], linewidth=line_width, label='TS (Correlated)')
    plt.plot(all_results.iters[0], bucb_corr_normalized_reward, c=colors[4], linewidth=line_width, label='BUCB (Correlated)')

    for ts_corr_prior, color, label in zip(ts_corr_prior_normalized_reward, colors[5:5+len(ts_corr_prior_normalized_reward)],
                                           config['priors_feature_names']):
        plt.plot(all_results.iters[0], ts_corr_prior,
                 c=color, linewidth=line_width, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))

    for bucb_corr_prior, color, label in zip(bucb_corr_prior_normalized_reward, colors[5+len(ts_corr_prior_normalized_reward):],
                                           config['priors_feature_names']):
        plt.plot(all_results.iters[0], bucb_corr_prior,
                 c=color, linewidth=line_width, label='BUCB (%s)' %(label.replace('nearest_features', 'Priors')))

    plt.xlim(0, np.max(all_results.iters[0]))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    figname = 'avg_reward.pdf'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)

    # plot normalized rewards with error bars
    num_skip = 10
    plt.figure()
    plt.errorbar(all_results.iters[0][::num_skip], ua_normalized_reward[::num_skip], yerr=ua_normalized_std_reward[::num_skip], c=colors[0], linewidth=line_width, label='Uniform')
    plt.errorbar(all_results.iters[0][::num_skip], ts_normalized_reward[::num_skip], yerr=ts_normalized_std_reward[::num_skip], c=colors[1], linewidth=line_width, label='TS (Uncorrelated)')
    plt.errorbar(all_results.iters[0][::num_skip], gi_normalized_reward[::num_skip], yerr=gi_normalized_std_reward[::num_skip], c=colors[2], linewidth=line_width, label='Gittins Indices')
    plt.errorbar(all_results.iters[0][::num_skip], ts_corr_normalized_reward[::num_skip], yerr=ts_corr_normalized_std_reward[::num_skip], c=colors[3], linewidth=line_width, label='TS (Correlated)')
    plt.errorbar(all_results.iters[0][::num_skip], bucb_corr_normalized_reward[::num_skip], yerr=bucb_corr_normalized_std_reward[::num_skip], c=colors[4], linewidth=line_width, label='BUCB (Correlated)')

    for ts_corr_prior, ts_corr_prior_std, color, label in zip(ts_corr_prior_normalized_reward, ts_corr_prior_normalized_std_reward,
                                                              colors[5:5+len(ts_corr_prior_normalized_reward)],
                                                              config['priors_feature_names']):
        plt.errorbar(all_results.iters[0][::num_skip], ts_corr_prior[::num_skip], yerr=ts_corr_prior_std[::num_skip],
                     c=color, linewidth=line_width, label='TS (%s)' %(label.replace('nearest_features', 'Priors')))

    for bucb_corr_prior, bucb_corr_prior_std, color, label in zip(bucb_corr_prior_normalized_reward, bucb_corr_prior_normalized_std_reward,
                                                                  colors[5+len(ts_corr_prior_normalized_reward):],
                                                                  config['priors_feature_names']):
        plt.errorbar(all_results.iters[0][::num_skip], bucb_corr_prior[::num_skip], yerr=bucb_corr_prior_std[::num_skip],
                     c=color, linewidth=line_width, label='BUCB (%s)' %(label.replace('nearest_features', 'Priors')))

    plt.xlim(0, np.max(all_results.iters[0]))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    figname = 'avg_reward_with_std.pdf'
    plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
    logging.info('Finished plotting %s', figname)
