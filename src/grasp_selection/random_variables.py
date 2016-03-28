"""
Random Variables for sampling force closure, etc
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import copy
import itertools as it
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

import scipy.linalg
import scipy.stats
import sklearn.cluster

import antipodal_grasp_sampler as ags
import grasp as gr
import graspable_object as go
import obj_file
import quality as pgq
import sdf_file
import similarity_tf as stf
import tfx
import feature_functions as ff
import discrete_adaptive_samplers as das
import models
import objectives
import termination_conditions as tc

import IPython

# TODO: move this function somewhere interesting
def skew(xi):
    S = np.array([[0, -xi[2], xi[1]],
                  [xi[2], 0, -xi[0]],
                  [-xi[1], xi[0], 0]])
    return S

def deskew(S):
    x = np.zeros(3)
    x[0] = S[2,1]
    x[1] = S[0,2]
    x[2] = S[1,0]
    return x

class RandomVariable(object):
    __metaclass__ = ABCMeta

    def __init__(self, num_prealloc_samples=0):
        self.num_prealloc_samples_ = num_prealloc_samples
        if self.num_prealloc_samples_ > 0:
            self._preallocate_samples()

    def _preallocate_samples(self, size=1):
        """ Preallocate samples for faster adative sampling """
        self.prealloc_samples_ = []
        for i in range(self.num_prealloc_samples_):
            self.prealloc_samples_.append(self.sample())

    @abstractmethod
    def sample(self, size=1):
        """ Sample | size | random variables """
        pass

    def rvs(self, size=1, iteration=1):
        """ Sample |size| random variables with the option of using preallocated samples """
        if self.num_prealloc_samples_ > 0:
            samples = []
            for i in range(size):
                samples.append(self.prealloc_samples_[(iteration + i) % self.num_prealloc_samples_])
            if size == 1:
                return samples[0]
            return samples
        # generate a new sample
        return self.sample(size=size)

class ArtificialRV(RandomVariable):
    '''
    A fake RV that always returns the given object
    '''
    def __init__(self, obj, num_prealloc_samples=0):
        self.obj_ = obj
        super(ArtificialRV, self).__init__(num_prealloc_samples)

    def sample(self, size = 1):
        return [self.obj_] * size
        
class ArtificialSingleRV(ArtificialRV):

    def sample(self, size = None):
        return self.obj_
        
class GraspableObjectPoseGaussianRV(RandomVariable):
    def __init__(self, obj, config):
        self.obj_ = obj
        self._parse_config(config)

        self.s_rv_ = scipy.stats.norm(obj.tf.scale, self.sigma_scale_**2)
        self.t_rv_ = scipy.stats.multivariate_normal(obj.tf.translation, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        RandomVariable.__init__(self, config['num_prealloc_obj_samples'])

    def _parse_config(self, config):
        # NOTE: scale sigma only for now
        self.sigma_rot_ = config['sigma_rot_obj']
        self.sigma_trans_ = config['sigma_trans_obj']
        self.sigma_scale_ = config['sigma_scale_obj']

    @property
    def obj(self):
        return self.obj_

    def sigma_pts(self, L, alpha=1e-3, kappa=0, scale=7.5, use_cov=True):
        """ Returns a set of sigma points with corresponding weights """
        lambda_o = alpha**2 * (L + kappa) - L

        mu_s = self.obj_.tf.scale
        mu_t = self.obj_.tf.translation
        mu_r = self.obj_.tf.rotation
        sigma_pts = [self.obj_]
        sigma_weights = [self.s_rv_.pdf(mu_s) * self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(np.zeros(3))]#[lambda_o / (L + lambda_o)]

        std = 1.0
        if use_cov:
            std = self.s_rv_.std()
        s_i = mu_s + np.sqrt(scale) * std
        tf_i = stf.SimilarityTransform3D(tfx.transform(mu_r, mu_t), s_i)
        obj_i = self.obj_.transform(tf_i)
        w_i = self.s_rv_.pdf(s_i) * self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(np.zeros(3))
        sigma_pts.append(obj_i)
        sigma_weights.append(w_i)

        s_i = mu_s - np.sqrt(scale) * std
        tf_i = stf.SimilarityTransform3D(tfx.transform(mu_r, mu_t), s_i)
        obj_i = self.obj_.transform(tf_i)
        w_i = self.s_rv_.pdf(s_i) * self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(np.zeros(3))
        sigma_pts.append(obj_i)
        sigma_weights.append(w_i)

        S_t = np.eye(3)
        if use_cov:
            S_t = self.t_rv_.cov
        for i in range(3):
            t_i = mu_t + np.sqrt(scale) * np.sqrt(S_t[:,i])
            tf_i = stf.SimilarityTransform3D(tfx.transform(mu_r, t_i), mu_s)
            obj_i = self.obj_.transform(tf_i)
            w_i = self.s_rv_.pdf(mu_s) * self.t_rv_.pdf(t_i) * self.r_xi_rv_.pdf(np.zeros(3))
            sigma_pts.append(obj_i)
            sigma_weights.append(w_i)

            t_i = mu_t - np.sqrt(scale) * np.sqrt(S_t[:,i])
            tf_i = stf.SimilarityTransform3D(tfx.transform(mu_r, t_i), mu_s)
            obj_i = self.obj_.transform(tf_i)
            w_i = self.s_rv_.pdf(mu_s) * self.t_rv_.pdf(t_i) * self.r_xi_rv_.pdf(np.zeros(3))
            sigma_pts.append(obj_i)
            sigma_weights.append(w_i)

        S_r = np.eye(3)
        if use_cov:
            S_r = self.r_xi_rv_.cov
        for i in range(3):
            r_i = np.sqrt(scale) * np.sqrt(S_r[:,i])
            w_i = self.s_rv_.pdf(mu_s) * self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(r_i)
            r_i = scipy.linalg.expm(skew(r_i)).dot(mu_r)
            tf_i = stf.SimilarityTransform3D(tfx.transform(r_i, mu_t), mu_s)
            obj_i = self.obj_.transform(tf_i)
            sigma_pts.append(obj_i)
            sigma_weights.append(w_i)

            r_i = - np.sqrt(scale) * np.sqrt(S_r[:,i])
            w_i = self.s_rv_.pdf(mu_s) * self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(r_i)
            r_i = scipy.linalg.expm(skew(r_i)).dot(mu_r)
            tf_i = stf.SimilarityTransform3D(tfx.transform(r_i, mu_t), mu_s)
            obj_i = self.obj_.transform(tf_i)
            sigma_pts.append(obj_i)
            sigma_weights.append(w_i)

        return sigma_pts, sigma_weights

    def sample(self, size=1):
        """ Sample |size| random variables from the model """
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)
            R = scipy.linalg.expm(S_xi).dot(self.obj_.tf.rotation)
            s = self.s_rv_.rvs(size=1)[0]
            t = self.t_rv_.rvs(size=1)
            sample_tf = stf.SimilarityTransform3D(tfx.transform(R.T, t), s)

            # transform object by pose
            obj_sample = self.obj_.transform(sample_tf)
            samples.append(obj_sample)

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples

class ParallelJawGraspPoseGaussianRV(RandomVariable):
    def __init__(self, grasp, config):
        self.grasp_ = grasp
        self._parse_config(config)

        self.t_rv_ = scipy.stats.multivariate_normal(grasp.center, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        RandomVariable.__init__(self, config['num_prealloc_grasp_samples'])

    def _parse_config(self, config):
        # NOTE: scale sigma only for now
        self.sigma_rot_ = config['sigma_rot_grasp']
        self.sigma_trans_ = config['sigma_trans_grasp']

    @property
    def grasp(self):
        return self.grasp_

    def sigma_pts(self, L, alpha=1e-3, kappa=0, scale=7.5, use_cov=True):
        """ Return a list of sigma pts where L specifies the RV dimension. Assumes this RV is not correlated with the others. """
        lambda_g = alpha**2 * (L + kappa) - L

        mu_t = self.grasp_.center
        mu_v = self.grasp_.axis
        mu_g = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(mu_t, mu_v, self.grasp_.grasp_width))
        sigma_pts = [mu_g]
        sigma_weights = [self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(np.zeros(3))]
        
        S_t = np.eye(3)
        if use_cov:
            S_t = self.t_rv_.cov
        for i in range(3):
            t_i = mu_t + np.sqrt(scale) * np.sqrt(S_t[:,i])
            g_i = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(t_i, mu_v, self.grasp_.grasp_width))
            w_i = self.t_rv_.pdf(t_i) * self.r_xi_rv_.pdf(np.zeros(3))
            sigma_pts.append(g_i)
            sigma_weights.append(w_i)

            t_i = mu_t - np.sqrt(scale) * np.sqrt(S_t[:,i])
            g_i = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(t_i, mu_v, self.grasp_.grasp_width))
            w_i = self.t_rv_.pdf(t_i) * self.r_xi_rv_.pdf(np.zeros(3))
            sigma_pts.append(g_i)
            sigma_weights.append(w_i)

        S_v = np.eye(3)
        if use_cov:
            S_v = self.r_xi_rv_.cov
        for i in range(3):
            v_i = np.sqrt(scale) * np.sqrt(S_v[:,i])
            w_i = self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(v_i)
            v_i = scipy.linalg.expm(skew(v_i)).dot(mu_v)
            g_i = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(mu_t, v_i, self.grasp_.grasp_width))
            sigma_pts.append(g_i)
            sigma_weights.append(w_i)

            v_i = - np.sqrt(scale) * np.sqrt(S_v[:,i])
            w_i = self.t_rv_.pdf(mu_t) * self.r_xi_rv_.pdf(v_i)
            v_i = scipy.linalg.expm(skew(v_i)).dot(mu_v)
            g_i = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(mu_t, v_i, self.grasp_.grasp_width))
            sigma_pts.append(g_i)
            sigma_weights.append(w_i)

        return sigma_pts, sigma_weights        

    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)
            
            v = scipy.linalg.expm(S_xi).dot(self.grasp_.axis)
            t = self.t_rv_.rvs(size=1)

            # transform object by pose
            grasp_sample = gr.ParallelJawPtGrasp3D(gr.ParallelJawPtGrasp3D.configuration_from_params(t, v, self.grasp_.grasp_width))

            samples.append(grasp_sample)

        if size == 1:
            return samples[0]
        return samples

class WrenchGaussianRV(RandomVariable):
    def __init__(self, target_wrench, config, force_limits=None):
        self.target_wrench_ = target_wrench
        self._parse_config(config)

        self.force_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.params_forces_**2)
        self.torque_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.params_torques_**2)
        self.force_limits_rv_ = scipy.stats.norm(0, self.params_scale_**2)
	if target_wrench is not None:
            self.force_rv_ = scipy.stats.multivariate_normal(self.target_wrench_[:3], self.params_forces_**2) #figure out forces
            self.torque_rv_ = scipy.stats.multivariate_normal(self.target_wrench_[3:], self.params_torques_**2) #figure out torques
        if force_limits is not None:
            self.force_limits_rv_ = scipy.stats.norm(self.params_['force_limits'], self.params_scale_**2)
            
        RandomVariable.__init__(self, config['num_prealloc_wrench_samples'])

    def _parse_config(self,config):
        self.params_forces_=config['sigma_forces_params']
        self.params_torques_=config['sigma_torques_params']
        self.params_scale_=config['sigma_scale_params']

    @property
    def params(self):
        return self.params_ 

    def sample(self,size=1):
        samples = []
        for i in range(size):
            # sample random force, torque, etc
            force = self.force_rv_.rvs(size=1)
            torque = self.torque_rv_.rvs(size=1)
            force_limits = self.force_limits_rv_.rvs(size=1)

            # create params dict for use in quality metrics
            wrench_sample = np.vstack((force, torque))
            params_dict_sample = {'force_limits':force_limits,'target_wrench':wrench_sample}
            samples.append(params_dict_sample)

        if size == 1:
            return samples[0]
        return samples

class FrictionGaussianRV(RandomVariable):
    def __init__(self, mu, sigma, config):
        self.friction_rv_ = scipy.stats.norm(mu, sigma)
        RandomVariable.__init__(self, config['num_prealloc_obj_samples'])

    def mean(self):
        return self.friction_rv_.mean()

    def sigma_pts(self, L, alpha=1e-3, kappa=0, scale=7.5, use_cov=True):
        lambda_f = alpha**2 * (L + kappa) - L

        mu_f = self.friction_rv_.mean()
        sigma_pts = [mu_f]
        sigma_weights = [self.friction_rv_.pdf(mu_f)]
        
        std = 1
        if use_cov:
            self.friction_rv_.std()
        f_i = mu_f + np.sqrt(scale) * std 
        w_i = self.friction_rv_.pdf(f_i)
        sigma_pts.append(f_i)
        sigma_weights.append(w_i)

        f_i = mu_f - np.sqrt(scale) * std
        w_i = self.friction_rv_.pdf(f_i)
        sigma_pts.append(f_i)
        sigma_weights.append(w_i)

        return sigma_pts, sigma_weights
        
    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random force, torque, etc
            friction_sample = self.friction_rv_.rvs(size=1)
            samples.append(friction_sample)

        if size == 1:
            return samples[0]
        return samples

def plot_value_vs_time_beta_bernoulli(result, candidate_true_p, true_max=None, color='blue'):
    """ Plots the number of samples for each value in for a discrete adaptive sampler"""
    best_values = [candidate_true_p[m.best_pred_ind] for m in result.models]
    plt.plot(result.iters, best_values, color=color, linewidth=2)
    if true_max is not None: # also plot best possible
        plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')

def test_antipodal_grasp_thompson():
    np.random.seed(100)

    # load object
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = go.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    config = {
        'grasp_width': 0.1,
        'friction_coef': 0.5,
        'num_cone_faces': 8,
        'grasp_samples_per_surface_point': 4,
        'dir_prior': 1.0,
        'alpha_thresh_div': 32,
        'rho_thresh': 0.75, # as pct of object max moment
        'vis_antipodal': False,
        'min_num_grasps': 20,
        'alpha_inc': 1.1,
        'rho_inc': 1.1,
        'sigma_mu': 0.1,
        'sigma_trans_grasp': 0.001,
        'sigma_rot_grasp': 0.1,
        'sigma_trans_obj': 0.001,
        'sigma_rot_obj': 0.1,
        'sigma_scale_obj': 0.1,
        'num_prealloc_obj_samples': 100,
        'num_prealloc_grasp_samples': 0,
        'min_num_collision_free_grasps': 10,
        'grasp_theta_res': 1
    }
    sampler = ags.AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    # convert grasps to RVs for optimization
    graspable_rv = GraspableObjectGaussianPose(graspable, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
    candidates = []
    for grasp in grasps:
        grasp_rv = ParallelJawGraspGaussian(grasp, config)
        candidates.append(ForceClosureRV(grasp_rv, graspable_rv, f_rv, config))

    objective = objectives.RandomBinaryObjective()

    # run bandits
    eps = 5e-4
    ua_tc_list = [tc.MaxIterTerminationCondition(1000)]#, tc.ConfidenceTerminationCondition(eps)]
    ua = das.UniformAllocationMean(objective, candidates)
    ua_result = ua.solve(termination_condition = tc.OrTerminationCondition(ua_tc_list), snapshot_rate = 100)
    logging.info('Uniform allocation took %f sec' %(ua_result.total_time))

    ts_tc_list = [tc.MaxIterTerminationCondition(1000), tc.ConfidenceTerminationCondition(eps)]
    ts = das.ThompsonSampling(objective, candidates)
    ts_result = ts.solve(termination_condition = tc.OrTerminationCondition(ts_tc_list), snapshot_rate = 100)
    logging.info('Thompson sampling took %f sec' %(ts_result.total_time))

    true_means = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)

    # plot results
    plt.figure()
    plot_value_vs_time_beta_bernoulli(ua_result, true_means, color='red')
    plot_value_vs_time_beta_bernoulli(ts_result, true_means, color='blue')
    plt.show()

    das.plot_num_pulls_beta_bernoulli(ua_result)
    plt.title('Observations Per Variable for Uniform allocation')

    das.plot_num_pulls_beta_bernoulli(ts_result)
    plt.title('Observations Per Variable for Thompson sampling')

    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_antipodal_grasp_thompson()
