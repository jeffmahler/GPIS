"""
Generates plots of the distribtions on the hyperparam penalty functions and computes relevant statistics
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
import operator
import os
import sys
import scipy.spatial.distance as ssd

import hyperparam_evaluation as he
from hyperparam_evaluation import HyperparamEvalResult
import experiment_config as ec

def hyperparam_signature(hyperparams):
    return 'grad_%s_moment_%f_shape_%f_nn_%d' %(hyperparams['weight_grad'], hyperparams['weight_moment'], hyperparams['weight_shape'], hyperparams['num_neighbors'])

class HyperparamStats:
    def __init__(self, hs, avg_ce, std_ce, med_ce,
                 avg_se, std_se, med_se,
                 avg_ccbp_ll, std_ccbp_ll, med_ccbp_ll):
        self.hs = hs
        
        self.avg_ce = avg_ce
        self.std_ce = std_ce
        self.med_ce = med_ce

        self.avg_se = avg_se
        self.std_se = std_se
        self.med_se = med_se

        self.avg_ccbp_ll = avg_ccbp_ll
        self.std_ccbp_ll = std_ccbp_ll
        self.med_ccbp_ll = med_ccbp_ll

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('result_dir')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    config = ec.ExperimentConfig(args.config)
    result_dir = args.result_dir

    hyperparam_results = {}

    # read in all pickle files
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
                            hs = hyperparam_signature(p.hyperparams) 
                            if hs not in hyperparam_results.keys():
                                hyperparam_results[hs] = [] 
                            hyperparam_results[hs].append(p)

    if len(hyperparam_results.keys()) == 0:
        exit(0)

    # plot params
    line_width = config['line_width']
    font_size = config['font_size']
    dpi = config['dpi']
    num_bins = 100

    hyperparam_stats = []

    # aggregate results for each set of hypers
    for hs, results in hyperparam_results.iteritems():
        # get all the values
        all_results = HyperparamEvalResult.compile_results(results)
        ce_vals = all_results.ce_vals[:,1]
        se_vals = all_results.se_vals[:,1]
        we_vals = all_results.we_vals[:,1]
        ccbp_ll_vals = all_results.ccbp_ll_vals[:,1]
        total_num_grasps = np.sum(all_results.num_grasps)
    
        # plot histograms of the metrics across all objects
        plotting.plot_histogram(ce_vals)
        plt.xlabel('Cross Entropy', fontsize=font_size)
        plt.ylabel('Num Objects', fontsize=font_size)
        plt.title('Histogram of Objects by Cross Entropy', fontsize=font_size)
        figname = '%s_ce_hist.pdf' %(hs)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        plotting.plot_histogram(se_vals)
        plt.xlabel('Squared Error', fontsize=font_size)
        plt.ylabel('Num Objects', fontsize=font_size)
        plt.title('Histogram of Objects by Squared Error', fontsize=font_size)
        figname = '%s_se_hist.pdf' %(hs)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        plotting.plot_histogram(we_vals)
        plt.xlabel('Weighted Squared Error', fontsize=font_size)
        plt.ylabel('Num Objects', fontsize=font_size)
        plt.title('Histogram of Objects by Weighted Squared Error', fontsize=font_size)
        figname = '%s_we_hist.pdf' %(hs)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        plotting.plot_histogram(ccbp_ll_vals)
        plt.xlabel('CCBP Log Likelihood', fontsize=font_size)
        plt.ylabel('Num Objects', fontsize=font_size)
        plt.title('Histogram of Objects by CCBP Log Likelihood', fontsize=font_size)
        figname = '%s_ccbp_ll_hist.pdf' %(hs)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)
        logging.info('Finished plotting %s', figname)

        # compute average std, and med vals per grasp 
        unnorm_ce = np.diag(all_results.num_grasps).dot(ce_vals)
        unnorm_se = np.diag(all_results.num_grasps).dot(se_vals)
        unnorm_ccbp_ll = np.diag(all_results.num_grasps).dot(ccbp_ll_vals)

        avg_ce = (1.0 / total_num_grasps) * np.sum(unnorm_ce, axis = 0) 
        avg_se = (1.0 / total_num_grasps) * np.sum(unnorm_se, axis = 0) 
        avg_ccbp_ll = (1.0 / total_num_grasps) * np.sum(unnorm_ccbp_ll, axis = 0) 

        std_ce = np.mean((ce_vals - avg_ce)**2, axis = 0) 
        std_se = np.mean((se_vals - avg_se)**2, axis = 0) 
        std_ccbp_ll = np.mean((ccbp_ll_vals - avg_ccbp_ll)**2, axis = 0) 

        med_ce = np.median(ce_vals, axis = 0) # not the real median but close enough
        med_se = np.median(se_vals, axis = 0) 
        med_ccbp_ll = np.median(ccbp_ll_vals, axis = 0)         

        logging.info('Hyperparams %s' %(hs))
        logging.info('CE Mean: %f Std: %f Med: %f' %(avg_ce, std_ce, med_ce))
        logging.info('SE Mean: %f Std: %f Med: %f' %(avg_se, std_se, med_se))
        logging.info('CCBP LL Mean: %f Std: %f Med: %f' %(avg_ccbp_ll, std_ccbp_ll, med_ccbp_ll))

        h = HyperparamStats(hs, avg_ce, std_ce, med_ce, avg_se, std_se, med_se, avg_ccbp_ll, std_ccbp_ll, med_ccbp_ll)
        hyperparam_stats.append(h)

    # find sort hyperparams by avgs
    ce_hypers = sorted(hyperparam_stats, key=operator.attrgetter('avg_ce'))
    se_hypers = sorted(hyperparam_stats, key=operator.attrgetter('avg_se'))
    ccbp_ll_hypers = sorted(hyperparam_stats, key=operator.attrgetter('avg_ccbp_ll'))

    logging.info('Max Avg CE was %f for hypers %s' %(ce_hypers[-1].avg_ce, ce_hypers[-1].hs))
    logging.info('Max Avg SE was %f for hypers %s' %(se_hypers[-1].avg_se, se_hypers[-1].hs))
    logging.info('Max Avg CCBP was %f for hypers %s' %(ccbp_ll_hypers[-1].avg_ccbp_ll, ccbp_ll_hypers[-1].hs))
    
