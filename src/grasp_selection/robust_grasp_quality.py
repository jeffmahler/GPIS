import copy
import itertools as it
<<<<<<< HEAD
import logging
=======
>>>>>>> dev
import numpy as np
import os
import sys
import time

import scipy.stats

import grasp as gr
import graspable_object as go
import obj_file
import quality as pgq
import random_variables as rvs
import sdf_file
import similarity_tf as stf
import tfx
import feature_functions as ff

import discrete_adaptive_samplers as das
import models
import objectives
import termination_conditions as tc

import IPython

class GraspQualityRV(rvs.RandomVariable):
    """ RV class for grasp quality on an object """
    def __init__(self, grasp_rv, obj_rv, friction_coef_rv, config, quality_metric="force_closure", params_rv=None):
        self.grasp_rv_ = grasp_rv
        self.obj_rv_ = obj_rv
        self.friction_coef_rv_ = friction_coef_rv # scipy stat rv
        self.params_rv_ = params_rv # samples extra params for quality

        self.features_ = None
        self.quality_metric_ = quality_metric

        self.sample_count_ = 0
        self.soft_fingers_ = True
        self._parse_config(config)

        # preallocation not available
        rvs.RandomVariable.__init__(self, num_prealloc_samples=0)

    def _parse_config(self, config):
        """ Grab config data from the config file """
        self.num_cone_faces_ = config['num_cone_faces']
        if config['use_soft_fingers']:
            self.soft_fingers_ = config['use_soft_fingers']

    @property
    def grasp(self):
        return self.grasp_rv_.grasp

    @property
    def features(self):
        if self.features_ is None:
            logging.warning('Features are uninitialized.')
        else:
            return self.features_.phi

    def set_features(self, features):
        self.features_ = features

    @property
    def feature_extractors(self):
        if self.features_ is None:
            logging.warning('Features are uninitialized.')
        else:
            return self.features_

    def vector_to_objects(self, x):
        if x.shape[0] != 14:
            raise ValueError('Must provide 14 d vector')
        g_center = x[:3]
        g_axis = x[3:6]
        o_scale = x[6]
        o_trans = x[7:10]
        o_rot = x[10:13]
        friction = x[13]
        o_rot = scipy.linalg.expm(rvs.skew(o_rot)).dot(self.obj_rv_.obj_.tf.rotation)
        tf = stf.SimilarityTransform3D(tfx.transform(o_rot, o_trans), o_scale)
        obj = self.obj_rv_.obj_.transform(tf)
        grasp = copy.copy(self.grasp_rv_.grasp)
        grasp.center_ = g_center
        grasp.axis_ = g_axis
        return grasp, obj, friction

    def objects_to_vector(self, grasp, obj, friction):
        x = np.zeros(14)
        x[:3] = grasp.center
        x[3:6] = grasp.axis
        x[6] = obj.tf.scale
        x[7:10] = obj.tf.translation
        x[10:13] = rvs.deskew(obj.tf.rotation)
        x[13] = friction
        return x

    def vec_quality(self, x):
        grasp, obj, friction = self.vector_to_objects(x)
        return pgq.PointGraspMetrics3D.grasp_quality(grasp, obj, self.quality_metric_, friction_coef = friction,
                                                     num_cone_faces = self.num_cone_faces_, soft_fingers = self.soft_fingers_, params = None)

    def grasp_covariance(self):
        dim = 14
        S = np.zeros([dim, dim])
        S[0:3,0:3] = self.grasp_rv_.t_rv_.cov
        S[3:6,3:6] = self.grasp_rv_.r_xi_rv_.cov
        S[6,6] = self.obj_rv_.s_rv_.std()**2
        S[7:10,7:10] = self.obj_rv_.t_rv_.cov
        S[10:13,10:13] = self.obj_rv_.r_xi_rv_.cov
        S[13,13] = self.friction_coef_rv_.friction_rv_.std()**2
        return S

    def taylor_approx_mean(self, scale=0.5):
        """ Taylor approx mean """
        dim = 14

        q_hess = np.zeros([dim, dim])
        grasp_vec = self.objects_to_vector(self.grasp_rv_.grasp_, self.obj_rv_.obj_, self.friction_coef_rv_.mean())
        q_mean = self.vec_quality(grasp_vec)
        S = scale * np.sqrt(np.diag(self.grasp_covariance()))
        for i in range(dim):
            dx = S[i]
            j = i
            if True:#for j in range(dim):
                dy = S[j]
                grasp_vec_di_dj = np.copy(grasp_vec)

                grasp_vec_dip_djm = np.copy(grasp_vec)
                grasp_vec_dim_djp = np.copy(grasp_vec)
                grasp_vec_dim_djm = np.copy(grasp_vec)

                grasp_vec_di_dj[i] += dx
                grasp_vec_di_dj[j] += dy

                grasp_vec_dip_djm[i] += dx
                grasp_vec_dip_djm[j] -= dy

                grasp_vec_dim_djp[i] -= dx
                grasp_vec_dim_djp[j] += dy

                grasp_vec_dim_djm[i] -= dx
                grasp_vec_dim_djm[j] -= dy

                q_ip_jp = self.vec_quality(grasp_vec_di_dj)
                q_ip_jm = self.vec_quality(grasp_vec_dip_djm)
                q_im_jp = self.vec_quality(grasp_vec_dim_djp)
                q_im_jm = self.vec_quality(grasp_vec_dim_djm)
                
                q_hess[i,j] = (q_ip_jp - q_ip_jm - q_im_jp + q_im_jm) / (4 * dx*dy)

        grasp_mean = grasp_vec.reshape((dim, 1))
        approx_q_bar = q_mean + 0.5 * np.trace(q_hess.dot(self.grasp_covariance()))
        return approx_q_bar

    def sigma_pts(self, alpha=1e-1, kappa=0, scale=7.5):
        """ Get the sigma points and probabilities """
        L = 6 + 7 + 1
        grasp_sigma_pts, grasp_sigma_weights = self.grasp_rv_.sigma_pts(L, alpha, kappa, scale=scale)
        obj_sigma_pts, obj_sigma_weights = self.obj_rv_.sigma_pts(L, alpha, kappa, scale=scale)
        friction_sigma_pts, friction_sigma_weights = self.friction_coef_rv_.sigma_pts(L, alpha, kappa, scale=scale)

        all_sigma_pts = [[grasp_sigma_pts[0], obj_sigma_pts[0], friction_sigma_pts[0]]]
        all_sigma_weights = [grasp_sigma_weights[0] * obj_sigma_weights[0] * friction_sigma_weights[0]]

        for grasp_sigma_pt, grasp_sigma_weight in zip(grasp_sigma_pts[1:], grasp_sigma_weights[1:]):
            all_sigma_pts.append([grasp_sigma_pt, obj_sigma_pts[0], friction_sigma_pts[0]])
            all_sigma_weights.append(grasp_sigma_weight * obj_sigma_weights[0] * friction_sigma_weights[0])

        for obj_sigma_pt, obj_sigma_weight in zip(obj_sigma_pts[1:], obj_sigma_weights[1:]):
            all_sigma_pts.append([grasp_sigma_pts[0], obj_sigma_pt, friction_sigma_pts[0]])
            all_sigma_weights.append(grasp_sigma_weights[0] * obj_sigma_weight * friction_sigma_weights[0])

        for friction_sigma_pt, friction_sigma_weight in zip(friction_sigma_pts[1:], friction_sigma_weights[1:]):
            all_sigma_pts.append([grasp_sigma_pts[0], obj_sigma_pts[0], friction_sigma_pt])
            all_sigma_weights.append(grasp_sigma_weights[0] * obj_sigma_weights[0] * friction_sigma_weight)

        quality_sigma_pts = []
        for i, sigma_pt in enumerate(all_sigma_pts):
            grasp = sigma_pt[0]
            obj = sigma_pt[1]
            friction = sigma_pt[2]
            params = None

            q = pgq.PointGraspMetrics3D.grasp_quality(grasp, obj, self.quality_metric_, friction_coef = friction,
                                                      num_cone_faces = self.num_cone_faces_, soft_fingers = self.soft_fingers_, params = params)
            quality_sigma_pts.append(q)

        return quality_sigma_pts, all_sigma_weights

    def sample(self, size=1):
        """ Samples force closure """
        # sample grasp
        cur_time = time.clock()
        grasp_sample = self.grasp_rv_.rvs(size=1, iteration=self.sample_count_)
        grasp_time = time.clock()

        # sample object
        obj_sample = self.obj_rv_.rvs(size=1, iteration=self.sample_count_)
        obj_time = time.clock()

        # sample friction cone
        friction_coef_sample = self.friction_coef_rv_.rvs(size=1)
        friction_time = time.clock()

        # sample params
        params_rv_sample = None
        if self.params_rv_ is not None:
            params_rv_sample = self.params_rv_.rvs(size=1, iteration=self.sample_count_)
            params_time = time.clock()

        # compute force closure
        q = pgq.PointGraspMetrics3D.grasp_quality(grasp_sample, obj_sample, self.quality_metric_, friction_coef = friction_coef_sample,
                                                  num_cone_faces = self.num_cone_faces_, soft_fingers = self.soft_fingers_,
                                                  params = params_rv_sample)
        quality_time = time.clock()

        #logging.info('Took %f sec to compute quality' %(quality_time - friction_time))

        self.sample_count_ = self.sample_count_ + 1
        return q

class RobustGraspQuality:
    """ Computes robust quality measures using brute force """

    @staticmethod
    def probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric="force_closure", params_rv=None, features=None,
                            num_samples = 100, compute_variance=False):
        """
        Get the probability of success for a binary quality metric
        """
        # set up random variable
        q_rv = GraspQualityRV(grasp_rv, graspable_rv, f_rv, config, quality_metric=quality_metric, params_rv=params_rv)
        q_rv.set_features(features)
        candidates = [q_rv]

        # brute force with uniform allocation
        snapshot_rate = config['bandit_snapshot_rate']
        objective = objectives.RandomBinaryObjective()
        ua = das.UniformAllocationMean(objective, candidates)
        ua_result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(num_samples),
                             snapshot_rate = snapshot_rate)

        # convert to estimated prob success
        final_model = ua_result.models[-1]
        estimated_ps = models.BetaBernoulliModel.beta_mean(final_model.alphas, final_model.betas)
        if not compute_variance:
            return estimated_ps[0]
        estimated_var_ps = models.BetaBernoulliModel.sample_variance(final_model.alphas, final_model.betas)
        return estimated_ps[0], estimated_var_ps[0]

    @staticmethod
    def probability_success_sigma_pts(graspable_rv, grasp_rv, f_rv, config, quality_metric="force_closure", params_rv=None, scale=7.5):
        """
        Get the probability of success for a binary quality metric using sigma points instead of random sampling
        """
        # set up random variable
        q_rv = GraspQualityRV(grasp_rv, graspable_rv, f_rv, config, quality_metric=quality_metric, params_rv=params_rv)
        sigma_pts, sigma_weights = q_rv.sigma_pts(scale=scale)
        return np.sum(np.array(sigma_pts) * np.array(sigma_weights)) / np.sum(np.array(sigma_weights))

    @staticmethod
    def probability_success_taylor_approx(graspable_rv, grasp_rv, f_rv, config, quality_metric="force_closure", params_rv=None, scale=0.5):
        """
        Get the probability of success for a binary quality metric using sigma points instead of random sampling
        """
        # set up random variable
        q_rv = GraspQualityRV(grasp_rv, graspable_rv, f_rv, config, quality_metric=quality_metric, params_rv=params_rv)
        return q_rv.taylor_approx_mean(scale=scale)

    @staticmethod
    def expected_quality(graspable_rv, grasp_rv, f_rv, config, quality_metric="ferrari_canny_L1", params_rv=None, features=None,
                         num_samples = 100, compute_variance=False):
        """
        Get the probability of success for a binary quality metric
        """
        # set up random variable
        q_rv = GraspQualityRV(grasp_rv, graspable_rv, f_rv, config, quality_metric=quality_metric, params_rv=params_rv)
        q_rv.set_features(features)
        candidates = [q_rv]

        # brute force with uniform allocation
        snapshot_rate = config['bandit_snapshot_rate']
        objective = objectives.RandomContinuousObjective()
        ua = das.GaussianUniformAllocationMean(objective, candidates, mean_prior=0.0, sigma=1e-2)
        ua_result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(num_samples),
                             snapshot_rate = snapshot_rate)

        # convert to estimated prob success
        final_model = ua_result.models[-1]
        mn_q = final_model.means
        if not compute_variance:
            return mn_q[0]
        var_q = models.BetaBernoulliModel.sample_variance(final_model.alphas, final_model.betas)
        return mn_q, var_q
        
