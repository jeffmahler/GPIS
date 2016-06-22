import argparse
import copy
import IPython
import logging
import numpy as np
import os
import sys
import time

import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.stats as ss

import database as db
import experiment_config as ec
import quality
import random_variables as rvs
import robust_grasp_quality as rgq

class ParamSweepVar:
    def __init__(self, name, low_val, high_val, inc):
        self.name = name
        self.low = low_val
        self.high = high_val
        self.inc = inc
        self.val = low_val

class AdditiveParamSweepVar(ParamSweepVar):
    def __init__(self, name, low_val, high_val, inc):
        ParamSweepVar.__init__(self, name, low_val, high_val, inc)

    def __iter__(self):
        self.val = self.low
        return self

    def next(self):
        val = self.val
        if self.val > self.high:
            raise StopIteration
        else:
            self.val = self.val + self.inc
        return val

class MultiplicativeParamSweepVar(ParamSweepVar):
    def __init__(self, name, low_val, high_val, inc):
        ParamSweepVar.__init__(self, name, low_val, high_val, inc)

    def __iter__(self):
        self.val = self.low
        return self

    def next(self):
        val = self.val
        if self.val > self.high:
            raise StopIteration
        else:
            self.val = self.val * self.inc
        return val

class ParamSweepFactory:
    @staticmethod
    def create_var(name, config):
        var = None
        if config['type'] == 'add':
            var = AdditiveParamSweepVar(name, config['low'], config['high'],
                                        config['inc'])
        elif config['type'] == 'mult':
            var = MultiplicativeParamSweepVar(name, config['low'], config['high'],
                                              config['inc'])
        return var

def create_quality_configs(config, sweep_var, stable_pose=None):
    non_param_keys = ['friction_coef', 'num_cone_faces', 'soft_fingers',
                      'sigma_rot_grasp', 'sigma_trans_grasp',
                      'sigma_rot_obj', 'sigma_trans_obj',
                      'sigma_scale_obj', 'sigma_mu', 'num_samples',
                      'num_prealloc_obj_samples', 'num_prealloc_grasp_samples',
                      'bandit_snapshot_rate']
                      
    quality_configs = []
    for val in sweep_var:
        tmp_config = copy.deepcopy(config['quality'])
        tmp_config[sweep_var.name] = val
        quality_config = {'params':{}, 'R_sample_sigma': None}
        if stable_pose is not None:
            gravity_resist_force = quality.GRAVITY_ACCEL * config['mass'] * stable_pose.r[2,:]
            quality_config['params']['target_wrench'] = \
                np.array([gravity_resist_force, np.zeros(3)]).ravel()
        for key, val in tmp_config.iteritems():
            if key in non_param_keys:
                quality_config[key] = val
            else:
                quality_config['params'][key] = val
        quality_configs.append(quality_config)
    return quality_configs

def sweep_metric(metric_name, obj, grasps, stable_pose, sweep_var, metrics, config):
    """ Sweep a metric """
    quality_configs = create_quality_configs(config, sweep_var, stable_pose)
    for sweep_val, quality_config in zip(sweep_var, quality_configs):
        logging.info('Evaluating %s at %.4f' %(sweep_var.name, sweep_val))
        for grasp in grasps:
            grasp_tag = '%s_%d' %(obj.key, grasp.grasp_id)
            if grasp_tag not in metrics.keys():
                metrics[grasp_tag] = []
            q = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, metric_name,
                                                          friction_coef=quality_config['friction_coef'],
                                                          num_cone_faces=quality_config['num_cone_faces'],
                                                          soft_fingers=quality_config['soft_fingers'],                                                          
                                                          params=quality_config['params'])
            metrics[grasp_tag].append(q)
    return metrics, quality_configs

def sweep_robust_metric(metric_name, obj, grasps, stable_pose, sweep_var, metrics, config):
    """ Sweep a metric """
    quality_configs = create_quality_configs(config, sweep_var, stable_pose)
    for sweep_val, quality_config in zip(sweep_var, quality_configs):
        logging.info('Evaluating robust %s at %.4f' %(sweep_var.name, sweep_val))
        graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, quality_config)
        f_rv = ss.norm(quality_config['friction_coef'], quality_config['sigma_mu'])
        for grasp in grasps:
            logging.info('Evaluating grasp %d of %d' %(grasp.grasp_id, len(grasps)))
            grasp_tag = '%s_%d' %(obj.key, grasp.grasp_id)
            if grasp_tag not in metrics.keys():
                metrics[grasp_tag] = []
            grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, quality_config)
            params_rv = rvs.ArtificialSingleRV(quality_config['params'])
            if metric_name == 'ferrari_canny_L1' or metric_name == 'wrench_resist_ratio':
                q = rgq.RobustGraspQuality.expected_quality(graspable_rv, grasp_rv, f_rv,
                                                            quality_config, quality_metric=metric_name,
                                                            params_rv=params_rv,
                                                            num_samples=quality_config['num_samples'])
            else:
                q = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv,
                                                               quality_config, quality_metric=metric_name,
                                                               params_rv=params_rv,
                                                               num_samples=quality_config['num_samples'],
                                                               compute_variance=False)
            metrics[grasp_tag].append(q)
    return metrics, quality_configs

def analyze_results(metrics, metric_name, sweep_var,
                    config, output_dir=None, robust=False):
    # Plotting params
    font_size = config['plotting']['font_size']
    dpi = config['plotting']['dpi']
    
    # Rankings
    grasp_metrics = {}
    for i, val in enumerate(sweep_var):
        grasp_metrics[val] = []
        for grasp_id, metric_vals in metrics.iteritems():
            grasp_metrics[val].append(metric_vals[i])

    # Analyze correlations and smoothness
    pr_corr_coefs = []
    sp_corr_coefs = []
    p_vals = []
    avg_diffs = []
    for param_val1, metric_list1 in grasp_metrics.iteritems():
        pr_corr_coefs.append([])
        sp_corr_coefs.append([])
        p_vals.append([])
        avg_diffs.append([])
        for param_val2, metric_list2 in grasp_metrics.iteritems():
            # correlations
            rho, p = ss.pearsonr(metric_list1, metric_list2)
            pr_corr_coefs[-1].append(rho)
            rho, p = ss.spearmanr(metric_list1, metric_list2)
            if np.isnan(rho):
                rho = 0
            sp_corr_coefs[-1].append(rho)
            p_vals[-1].append(p)

            # smoothness
            metric_diffs = np.array(metric_list1) - np.array(metric_list2)
            param_diff = param_val1 - param_val2
            avg_diff = 0.0
            if param_diff != 0.0:
                avg_diff = np.mean(metric_diffs) / param_diff
            avg_diffs[-1].append(avg_diff)

    # convert to numpy
    sp_corr_coefs = np.array(sp_corr_coefs)
    pr_corr_coefs = np.array(pr_corr_coefs)
    p_vals = np.array(p_vals)
    avg_diffs = np.array(avg_diffs)

    # plot correlation matrix
    clim = (-1, 1)
    plt.figure()
    plt.title('Metric %s Param %s' %(metric_name, sweep_var.name), fontsize=font_size)
    plt.subplot(1,2,1)
    plt.imshow(sp_corr_coefs, interpolation='none', cmap=plt.cm.gray, clim=clim)
    plt.colorbar()
    plt.title('Spearman Coefficients', fontsize=font_size)
    plt.subplot(1,2,2)
    plt.imshow(pr_corr_coefs, interpolation='none', cmap=plt.cm.gray, clim=clim)
    plt.colorbar()
    plt.title('Pearson Coefficients', fontsize=font_size)

    if output_dir is not None:
        if robust:
            figname = os.path.join(output_dir,
                                   'robust_%s_param_%s_correlations.pdf' %(metric_name, sweep_var.name))
        else:
            figname = os.path.join(output_dir,
                                   'metric_%s_param_%s_correlations.pdf' %(metric_name, sweep_var.name))            
        plt.savefig(figname, dpi=dpi)

    # plot smoothness
    plt.figure()
    plt.title('Metric %s Param %s' %(metric_name, sweep_var.name), fontsize=font_size)
    plt.imshow(avg_diffs, interpolation='none', cmap=plt.cm.gray)
    plt.colorbar()
    plt.title('Smoothness estimates', fontsize=font_size)    
    if output_dir is not None:
        if robust:
            figname = os.path.join(output_dir,
                                   'robust_%s_param_%s_smoothness.pdf' %(metric_name, sweep_var.name))
        else:
            figname = os.path.join(output_dir,
                                   'metric_%s_param_%s_smoothness.pdf' %(metric_name, sweep_var.name))            
        plt.savefig(figname, dpi=dpi)

    # estimate lipschitz
    n = avg_diffs.shape[0]
    lipschitz_est = np.sum(avg_diffs) / (n**2 - n)
    logging.info('Lipschitz estimate for metric %s wrt parameter %s: %.4f' %(metric_name, sweep_var.name, lipschitz_est)) 

    if output_dir is not None:
        pass
    else:
        plt.show()

    return sp_corr_coefs, pr_corr_coefs, p_vals, avg_diffs, lipschitz_est

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    # read config file and open databse
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    dataset = database.dataset(config['dataset'])

    # setup output folder
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create sweep vars for robust metrics
    for metric_name, metric_sweep_config in config['robust_metrics'].iteritems():
        logging.info('Estimating sensitivity for robust metric %s' %(metric_name))
        for var_name, var_config in metric_sweep_config.iteritems():
            metrics = {}

            # sweep over metric values
            logging.info('Sweeping value %s' %(var_name))
            sweep_var = ParamSweepFactory.create_var(var_name, var_config)

            # compute metrics for each object
            for obj in dataset:
                logging.info('Computing robust metrics for object %s' %(obj.key))
                grasps = dataset.grasps(obj.key, gripper=config['gripper'])
                stable_pose = dataset.stable_pose(obj.key, config['ppc_stp_ids'][obj.key])
                metrics, quality_configs = \
                    sweep_robust_metric(metric_name, obj, grasps, stable_pose,
                                        sweep_var, metrics, config)
            
            # analyze results
            sp_corr_coefs, pr_corr_coefs, p_vals, avg_diffs, lipschitz_est = \
                analyze_results(metrics, metric_name, sweep_var,
                                config, output_dir=output_dir, robust=True)

            # save to output folder
            filename = os.path.join(output_dir,
                                    'robust_%s_param_%s_sp.npy' %(metric_name, var_name))
            np.save(filename, sp_corr_coefs)
            filename = os.path.join(output_dir,
                                    'robust_%s_param_%s_pr.npy' %(metric_name, var_name))
            np.save(filename, pr_corr_coefs)
            filename = os.path.join(output_dir,
                                    'robust_%s_param_%s_pvals.npy' %(metric_name, var_name))
            np.save(filename, p_vals)
            filename = os.path.join(output_dir,
                                    'robust_%s_param_%s_diffs.npy' %(metric_name, var_name))
            np.save(filename, avg_diffs)
            filename = os.path.join(output_dir,
                                    'robust_%s_param_%s_lip.npy' %(metric_name, var_name))
            np.save(filename, np.array(lipschitz_est))
    
    # create sweep vars for deterministic metrics
    for metric_name, metric_sweep_config in config['metrics'].iteritems():
        logging.info('Estimating sensitivity for metric %s' %(metric_name))
        for var_name, var_config in metric_sweep_config.iteritems():
            metrics = {}

            # sweep over metric values
            logging.info('Sweeping value %s' %(var_name))
            sweep_var = ParamSweepFactory.create_var(var_name, var_config)

            # compute metrics for each object
            for obj in dataset:
                logging.info('Computing metrics for object %s' %(obj.key))
                grasps = dataset.grasps(obj.key, gripper=config['gripper'])
                stable_pose = dataset.stable_pose(obj.key, config['ppc_stp_ids'][obj.key])
                metrics, quality_configs = \
                    sweep_metric(metric_name, obj, grasps, stable_pose,
                                 sweep_var, metrics, config)
            
            # analyze results
            sp_corr_coefs, pr_corr_coefs, p_vals, avg_diffs, lipschitz_est = \
                analyze_results(metrics, metric_name, sweep_var,
                                config, output_dir)

            # save to output folder
            filename = os.path.join(output_dir,
                                    'metric_%s_param_%s_sp.npy' %(metric_name, var_name))
            np.save(filename, sp_corr_coefs)
            filename = os.path.join(output_dir,
                                    'metric_%s_param_%s_pr.npy' %(metric_name, var_name))
            np.save(filename, pr_corr_coefs)
            filename = os.path.join(output_dir,
                                    'metric_%s_param_%s_pvals.npy' %(metric_name, var_name))
            np.save(filename, p_vals)
            filename = os.path.join(output_dir,
                                    'metric_%s_param_%s_diffs.npy' %(metric_name, var_name))
            np.save(filename, avg_diffs)
            filename = os.path.join(output_dir,
                                    'metric_%s_param_%s_lip.npy' %(metric_name, var_name))
            np.save(filename, np.array(lipschitz_est))
