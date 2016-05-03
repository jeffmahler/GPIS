import copy
import IPython
import logging
import numpy as np
import os
import sys
import scipy.stats

import matplotlib.pyplot as plt
import scipy.stats as ss

import database as db
import experiment_config as ec
import objectives
import random_variables as rvs
import robust_grasp_quality as rgq

GRAVITY_ACCEL = 9.81

class PartialClosureInput:
    def __init__(self, graspable, grasp):
        self.graspable_ = graspable
        self.grasp_ = grasp

    @property
    def graspable(self):
        return self.graspable_

    @property
    def grasp(self):
        return self.grasp_

class PartialClosureParams(objectives.FiniteDiffableVariable):
    def __init__(self, mass_dict, force_limits, friction_coef, sigma_mu, sigma_rot_obj, sigma_trans_obj,
                 sigma_rot_grasp, sigma_trans_grasp):
        self.mass_dict_ = mass_dict
        self.force_limits_ = force_limits
        self.friction_coef_ = friction_coef
        self.sigma_mu_ = sigma_mu
        self.sigma_rot_obj_ = sigma_rot_obj
        self.sigma_trans_obj_ = sigma_trans_obj
        self.sigma_rot_grasp_ = sigma_rot_grasp
        self.sigma_trans_grasp_ = sigma_trans_grasp

        self.num_dimensions_ = len(self.mass_dict_.keys()) + 7

    def num_dimensions(self):
        return self.num_dimensions_

    def add(self, delta, i):
        if i >= self.num_dimensions:
            raise ValueError('Dimension %d too large' %(dim))

        other_params = copy.deepcopy(self)

        if i < len(self.mass_dict_.keys()):
            other_params.mass_dict_[self.mass_dict_.keys()[i]] += delta
        elif i == 8:
            other_params.force_limits_ += delta
        elif i == 9:
            other_params.friction_coef_ += delta
        elif i == 10:
            other_params.sigma_mu_ += delta
        elif i == 11:
            other_params.sigma_trans_obj_ += delta
        elif i == 12:
            other_params.sigma_rot_obj_ += delta
        elif i == 13:
            other_params.sigma_trans_grasp_ += delta
        elif i == 14:
            other_params.sigma_rot_grasp_ += delta
        return other_params

class PartialClosureFunc:
    def __call__(self, x, theta, params):
        """Evaluate partial closure given all parameters."""
        graspable = x.graspable
        grasp = x.grasp
    
        config = {
            # object uncertainty
            'sigma_rot_obj': theta.sigma_rot_obj_,
            'sigma_trans_obj': theta.sigma_trans_obj_,
            'sigma_scale_obj': 0,

            # gripper uncertainty
            'sigma_rot_grasp': theta.sigma_rot_grasp_,
            'sigma_trans_grasp': theta.sigma_trans_grasp_,

            # friction uncertainty
            'friction_coef': theta.friction_coef_,
            'sigma_mu': theta.sigma_mu_,

            # other parameters
            'num_cone_faces': params['num_cone_faces'],
            'use_soft_fingers': params['use_soft_fingers'],
            'bandit_snapshot_rate': params['bandit_snapshot_rate'],
            'ppc_num_samples': params['ppc_num_samples'],
            'num_prealloc_obj_samples': params['num_prealloc_obj_samples'],
            'num_prealloc_grasp_samples': params['num_prealloc_grasp_samples'],
            }

        stable_pose_normal = params['stable_poses'][graspable.key].r[2]
        gravity_resist_force =  GRAVITY_ACCEL * theta.mass_dict_[graspable.key] * stable_pose_normal

        graspable_rv = rvs.GraspableObjectPoseGaussianRV(graspable, config)
        grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, config)
        f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
        params_rv = rvs.ArtificialSingleRV({
                'force_limits': theta.force_limits_,
                'target_wrench': np.append(gravity_resist_force, [0, 0, 0]), # positive because we are exerting the wrench
                })

        ppc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv,
                                                         config, quality_metric='partial_closure',
                                                         params_rv=params_rv,
                                                         num_samples=config['ppc_num_samples'],
                                                         compute_variance=False)
        return ppc

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    result_dir = sys.argv[1]
    output_dir = sys.argv[2]
    config_filename = sys.argv[3]

    success_metric_key = 'p_grasp_success'
    lift_metric_key = 'p_lift_success'
    grasp_id_name = 'grasp_id'
    grasp_result_file_root = 'grasp_metric_results.csv'
    font_size = 15
    marker_size = 100
    eps = 0
    dpi = 400
    a = 13

    all_grasp_metric_data = {}
    X = []
    Y = []
    mass_dict = {}

    # open up database
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    database = db.Hdf5Database(database_filename, config)
    dataset = database.dataset(dataset_name)

    params = {}
    params['num_cone_faces'] = config['num_cone_faces']
    params['use_soft_fingers'] = config['use_soft_fingers']
    params['bandit_snapshot_rate'] = config['bandit_snapshot_rate']
    params['ppc_num_samples'] = config['ppc_num_samples']
    params['num_prealloc_obj_samples'] = config['num_prealloc_obj_samples']
    params['num_prealloc_grasp_samples'] = config['num_prealloc_grasp_samples']
    params['stable_poses'] = {}

    # extract all grasp metric data
    for experiment_dir in os.listdir(result_dir):
        grasp_result_filename = os.path.join(result_dir, experiment_dir, grasp_result_file_root)
        logging.info('Reading result %s' %(grasp_result_filename))
        try:
            grasp_metric_data = np.genfromtxt(grasp_result_filename, dtype=float, delimiter=',', names=True)
        except:
            grasp_metric_data = np.genfromtxt(grasp_result_filename, dtype=float, delimiter=';', names=True)            

        exp_config_filename = os.path.join(result_dir, experiment_dir, 'dexnet_physical_experiments.yaml')
        exp_config = ec.ExperimentConfig(exp_config_filename)
        object_name = exp_config['object_name']
        obj = dataset[object_name]
        stable_pose = dataset.stable_pose(object_name, 'pose_' + str(exp_config['stable_pose_index']))

        if object_name not in params['stable_poses'].keys():
            params['stable_poses'][object_name] = stable_pose

        if object_name not in mass_dict.keys():
            mass_dict[object_name] = config['object_mass']

        grasps = dataset.grasps(object_name, gripper=exp_config['gripper'])
        grasp_id_map = {}
        [grasp_id_map.update({g.grasp_id: g}) for g in grasps]

        column_names = grasp_metric_data.dtype.names
        for metric_name in column_names:
            if metric_name == grasp_id_name:
                for grasp_id in grasp_metric_data[metric_name].tolist():
                    X.append(PartialClosureInput(obj, grasp_id_map[grasp_id]))
            elif metric_name != 'ferrari_canny_l1' and \
                    metric_name != 'force_closure' and metric_name.find('vfc') == -1 and \
                    metric_name.find('vpc') == -1:
                metric_key = metric_name
                if metric_name.find('ppc') != -1:
                    metric_key = metric_name[:4] + metric_name[11:]
                if metric_name.find('lift_closure') != -1:
                    metric_key = 'lift_closure'

                # add to dict if nonexistent
                if metric_key not in all_grasp_metric_data.keys():
                    all_grasp_metric_data[metric_key] = []

                # add metrics
                all_grasp_metric_data[metric_key].extend(grasp_metric_data[metric_name].tolist())

            if metric_name == success_metric_key:
                Y.extend(grasp_metric_data[metric_name].tolist())

    """
    theta = PartialClosureParams(mass_dict, config['grasp_force_limit'], 0.5,
                                 0.1, 0.1, 0.005, 0.1, 0.005)
    obj = objectives.RidgeRegressionObjective(PartialClosureFunc(), X, Y, params=params)
    print obj(theta)
    IPython.embed()
    """

    # analyze correlations
    target_metrics = [success_metric_key, lift_metric_key]
    for target_metric in target_metrics:
        target_values = all_grasp_metric_data[target_metric]
        #target_values = [1 * (v > 0.5) for v in target_values]

        # check against all other metrics
        pr_corr_coefs = []
        sp_corr_coefs = []
        for metric_key in all_grasp_metric_data.keys():
            if metric_key != target_metric:
                corr_values = all_grasp_metric_data[metric_key]

                # compute correlation
                rho = np.corrcoef(target_values, corr_values)
                rho = rho[1,0]
                pr_corr_coefs.append((metric_key, rho))

                nu = ss.spearmanr(target_values, corr_values)
                sp_corr_coefs.append((metric_key, nu[0]))

                # scatter data
                plt.figure()
                plt.scatter(corr_values, target_values, color='b', s=marker_size)
                plt.xlim(-eps, np.max(np.array(corr_values))+eps)
                plt.ylim(-eps, 1+eps)
                plt.xlabel(metric_key[:a], fontsize=font_size)                
                plt.ylabel(target_metric, fontsize=font_size)
                plt.title('Correlation = %.3f' %(nu[0]), fontsize=font_size)

                figname = os.path.join(output_dir, '%s_vs_%s.pdf' %(target_metric, metric_key[:a]))
                plt.savefig(figname, dpi=dpi)

        # sort corr coefs and store
        pr_corr_coefs.sort(key = lambda x: x[1], reverse=True)
        sp_corr_coefs.sort(key = lambda x: x[1], reverse=True)

        # attempt to regress
        
        Y = target_values

                
