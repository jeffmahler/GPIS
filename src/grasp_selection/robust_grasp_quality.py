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
        self.sample_count_ = self.sample_count_ + 1
        return q

class RobustGraspQuality:
    """ Computes robust quality measures using brute force """

    @staticmethod
    def probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric="force_closure", params_rv=None, features=None,
                            num_samples = 100):
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
        return estimated_ps[0]

    @staticmethod
    def expected_quality(graspable_rv, grasp_rv, f_rv, config, quality_metric="ferrari_canny_L1", params_rv=None, features=None,
                         num_samples = 100):
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
        expected_q = final_model.means
        return expected_q[0]
        
