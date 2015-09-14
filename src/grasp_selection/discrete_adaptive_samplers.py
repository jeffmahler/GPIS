# -*- coding: utf-8 -*-

"""
Classes for selecting a candidate that maximizes some objective over a discrete set of candidates

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats
import time

import discrete_selection_policies as dcsp
import kernels
import models
import objectives
import solvers
import termination_conditions as tc

import IPython

class AdaptiveSamplingResult:
    """
    Basically a struct to store the results of sampling / optimization
    """
    def __init__(self, best_candidates, best_pred_means, best_pred_vars, total_time, checkpt_times, iters, indices, vals, models):
        self.best_candidates = best_candidates
        self.best_pred_means = best_pred_means
        self.best_pred_vars = best_pred_vars
        self.total_time = total_time
        self.checkpt_times = checkpt_times
        self.iters = iters
        self.indices = indices
        self.vals = vals
        self.models = models
        self.best_pred_ind = [m.best_pred_ind for m in models]

    def shrink(self):
        self.models = self.models[-1:]

class DiscreteAdaptiveSampler(solvers.DiscreteSamplingSolver):
    __metaclass__ = ABCMeta

    """
    Skeleton class for an adaptive sampler to maximize some objective over some objective.
    NOTE: Should NOT be instantiated directly. Always use a subclass that fixes the model and selection policy
    """
    def __init__(self, objective, candidates, model, selection_policy):
        self.model_ = model
        self.selection_policy_ = selection_policy
        solvers.DiscreteSamplingSolver.__init__(self, objective, candidates)

    @abstractmethod
    def reset_model(self, candidates):
        """ Reset model with the new candidates """
        # TODO: this feels a little hacky to me, but maybe we can make it work down the road
        pass

    def discrete_maximize(self, candidates, termination_condition = tc.MaxIterTerminationCondition(solvers.DEF_MAX_ITER),
                          snapshot_rate = 1):
        """
        Maximizes a function over a discrete set of variables by
        iteratively predicting the best point (using some model policy)
        """
        # check input
        if len(candidates) == 0:
            raise ValueError('No candidates specified')

        if not isinstance(self.model_, models.DiscreteModel):
            logging.error('Illegal model specified')
            raise ValueError('Illegitimate model used in DiscreteAdaptiveSampler')

        # init vars
        terminate = False
        k = 0 # cur iter
        num_candidates = len(candidates)
        self.reset_model(candidates) # update model with new candidates

        # logging
        times = []
        iters = []
        iter_indices = []
        iter_vals = []
        iter_models = []
        start_time = time.clock()
        next_ind_val = 0

        while not terminate:
            # get next point to sample
            next_ind = self.selection_policy_.choose_next()

            # evaluate the function at the given point (can be nondeterministic)
            prev_ind_val = next_ind_val
            next_ind_val = self.objective_.evaluate(candidates[next_ind])

            # snapshot the model and whatnot
            if (k % snapshot_rate) == 0:
                #logging.info('Iteration %d' %(k))

                # log time and stuff
                checkpt = time.clock()
                times.append(checkpt - start_time)
                iters.append(k)
                iter_indices.append(next_ind)
                iter_vals.append(next_ind_val)
                iter_models.append(self.model_.snapshot())

            # update the model (e.g. posterior update, grasp pruning)
            self.model_.update(next_ind, next_ind_val)

            # check termination condiation
            terminate = termination_condition(k, cur_val = next_ind_val, prev_val = prev_ind_val, model = self.model_)
            k = k + 1

        # log final values
        checkpt = time.clock()
        times.append(checkpt - start_time)
        iters.append(k)
        iter_indices.append(next_ind)
        iter_vals.append(next_ind_val)
        iter_models.append(self.model_.snapshot())

        # log total runtime
        end_time = time.clock()
        total_duration = end_time - start_time

        # log results and return
        best_indices, best_pred_means, best_pred_vars = self.model_.max_prediction()
        best_candidates = []
        num_best = best_indices.shape[0]
        for i in range(num_best):
            best_candidates.append(best_indices[i])
        return AdaptiveSamplingResult(best_candidates, best_pred_means, best_pred_vars, total_duration,
                                      times, iters, iter_indices, iter_vals, iter_models)


# Beta-Bernoulli bandit models: so easy!
class BetaBernoulliBandit(DiscreteAdaptiveSampler):
    """ Performs uniform allocation to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, policy, alpha_prior = 1.0, beta_prior = 1.0):
        self.num_candidates_ = len(candidates)
        self.model_ = models.BetaBernoulliModel(self.num_candidates_, alpha_prior, beta_prior)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        """ Needed to independently maximize over subsets of data """
        num_subcandidates = len(candidates)
        self.model_ = models.BetaBernoulliModel(self.num_candidates_, self.model_.alpha_prior_, self.model_.beta_prior_)
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class UniformAllocationMean(BetaBernoulliBandit):
    """ Performs uniform allocation to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        self.selection_policy_ = dcsp.UniformSelectionPolicy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

class ThompsonSampling(BetaBernoulliBandit):
    """ Performs thompson sampling to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        self.selection_policy_ = dcsp.ThompsonSelectionPolicy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

class GittinsIndex98(BetaBernoulliBandit):
    """ Performs Gittins index policy with gamma = 0.98 to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, alpha_prior = 1.0, beta_prior = 1.0):
        # NOTE: priors will be rounded to the nearest integers
        self.selection_policy_ = dcsp.BetaBernoulliGittinsIndex98Policy()
        BetaBernoulliBandit.__init__(self, objective, candidates, self.selection_policy_, alpha_prior, beta_prior)

# Gaussian bandit models
class GaussianBandit(DiscreteAdaptiveSampler):
    def __init__(self, objective, candidates, policy, mean_prior=0.5, sigma=1e-2):
        self.num_candidates_ = len(candidates)
        self.model_ = models.GaussianModel(self.num_candidates_, mean_prior, sigma)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        self.model_ = models.GaussianModel(self.num_candidates_, self.model_.mean_prior_, self.model_.sigma_)
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class GaussianUniformAllocationMean(GaussianBandit):
    def __init__(self, objective, candidates, mean_prior=0.5, sigma=1e-2):
        GaussianBandit.__init__(self, objective, candidates, dcsp.UniformSelectionPolicy(), mean_prior, sigma)

class GaussianThompsonSampling(GaussianBandit):
    def __init__(self, objective, candidates, mean_prior=0.5, sigma=1e-2):
        GaussianBandit.__init__(self, objective, candidates, dcsp.ThompsonSelectionPolicy(), mean_prior, sigma)

class GaussianUCBSampling(GaussianBandit):
    def __init__(self, objective, candidates, mean_prior=0.5, sigma=1e-2):
        GaussianBandit.__init__(self, objective, candidates, dcsp.GaussianUCBPolicy(), mean_prior, sigma)


# Correlated Beta-Bernoulli bandit models
class CorrelatedBetaBernoulliBandit(DiscreteAdaptiveSampler):
    """ Performs uniform allocation to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, policy, nn, kernel, tolerance=1e-4, alpha_prior=1.0, beta_prior=1.0, p=0.95):
        self.num_candidates_ = len(candidates)
        self.model_ = models.CorrelatedBetaBernoulliModel(candidates, nn, kernel, tolerance, alpha_prior, beta_prior, p)
        self.selection_policy_ = policy
        self.selection_policy_.set_model(self.model_)

        DiscreteAdaptiveSampler.__init__(self, objective, candidates, self.model_, self.selection_policy_)

    def reset_model(self, candidates):
        """ Needed to independently maximize over subsets of data """
        self.model_ = models.CorrelatedBetaBernoulliModel(
            self.candidates_, self.model_.nn_, self.model_.kernel_,
            self.model_.tolerance_, self.model_.alpha_prior_, self.model_.beta_prior_, p=self.model_.p_
        )
        self.selection_policy_.set_model(self.model_) # always update the selection policy!

class CorrelatedThompsonSampling(CorrelatedBetaBernoulliBandit):
    def __init__(self, objective, candidates, nn, kernel,
                 tolerance=1e-4, alpha_prior=1.0, beta_prior=1.0, p=0.95):
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, dcsp.ThompsonSelectionPolicy(),
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

class CorrelatedBayesUCB(CorrelatedBetaBernoulliBandit):
    """ Performs Gittins index policy with gamma = 0.98 to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, nn, kernel, tolerance=1e-4,
                 alpha_prior=1.0, beta_prior=1.0, horizon=1000, c=6, p=0.95):
        policy = dcsp.BetaBernoulliBayesUCBPolicy(horizon=horizon, c=c)
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, policy,
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

class CorrelatedGittins(CorrelatedBetaBernoulliBandit):
    """ Performs Gittins index policy with gamma = 0.98 to get the candidate that maximizes the mean value of the objective"""
    def __init__(self, objective, candidates, nn, kernel, tolerance=1e-4,
                 alpha_prior=1.0, beta_prior=1.0, p=0.95):
        policy = dcsp.BetaBernoulliGittinsIndex98Policy()
        CorrelatedBetaBernoulliBandit.__init__(
            self, objective, candidates, policy,
            nn, kernel, tolerance, alpha_prior, beta_prior, p
        )

class RandomVariable:
    """Abstract class for random variables."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample_success(self):
        pass

    @abstractmethod
    def value(self):
        pass

class BernoulliRV(RandomVariable):
    """ Bernoulli RV class for use with Beta-Bernoulli bandit testing """
    def __init__(self, p):
        self.p_ = p

    def sample_success(self):
        return scipy.stats.bernoulli.rvs(self.p_)

    def p(self):
        return self.p_

    def value(self):
        return self.p_

    def __repr__(self):
        return 'Bernoulli({})'.format(self.p_)

# Tests
NUM_CANDIDATES = 100
MAX_ITERS = 3000
SNAPSHOT_RATE = 100

def plot_num_pulls(result):
    """ Plots the number of samples for each value in for a discrete adaptive sampler"""
    num_candidates = result.models[-1].num_obs.shape[0]
    ind = np.arange(num_candidates)
    width = 1

    fig, ax = plt.subplots()
    ax.bar(ind+width, result.models[-1].num_obs)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Num observations')
    ax.set_xlim([0, NUM_CANDIDATES+1])

def plot_value_vs_time(result, candidates, true_max=None):
    """ Plots the number of samples for each value in for a discrete adaptive sampler"""
    best_values = [candidates[m.best_pred_ind].value() for m in result.models]
    plt.figure()
    plt.plot(result.iters, best_values, color='blue', linewidth=2)
    if true_max is not None: # also plot best possible
        plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')
    plt.xlim([0, MAX_ITERS+1])


def test_uniform_alloc(num_candidates=NUM_CANDIDATES):
    # get candidates
    np.random.seed(1000)
    pred_means = np.random.rand(num_candidates)
    candidates = []
    for i in range(num_candidates):
        candidates.append(BernoulliRV(pred_means[i]))

    # get true maximum
    true_max = np.max(pred_means)
    true_max_indices = np.where(pred_means == true_max)

    # solve using uniform allocation
    obj = objectives.RandomBinaryObjective()
    ua = UniformAllocationMean(obj, candidates)

    result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate = SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert(len(result.best_candidates) == 1)
    assert(np.abs(result.best_candidates[0].p() - true_max) < 1e-4)
    logging.info('Uniform alloc test passed!')
    logging.info('Took %f sec' %(result.total_time))
    logging.info('Best index %d' %(true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Uniform Allocation')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Uniform Allocation')

    return result

def test_thompson_sampling(num_candidates=NUM_CANDIDATES, random=False):
    # get candidates
    np.random.seed(1000)
    if random:
        pred_means = np.random.rand(num_candidates)
    else:
        pred_means = np.linspace(0.0, 1.0, num=num_candidates)
    candidates = []
    for i in range(num_candidates):
        candidates.append(BernoulliRV(pred_means[i]))

    # get true maximum
    true_max = np.max(pred_means)
    true_max_indices = np.where(pred_means == true_max)

    # solve using uniform allocation
    obj = objectives.RandomBinaryObjective()
    ua = ThompsonSampling(obj, candidates)

    result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate = SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert(len(result.best_candidates) == 1)
    assert(np.abs(result.best_candidates[0].p() - true_max) < 1e-4)
    logging.info('Thompson sampling test passed!')
    logging.info('Took %f sec' %(result.total_time))
    logging.info('Best index %d' %(true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Thompson Sampling')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Thompson Sampling')

    return result

def test_gittins_indices_98(num_candidates=NUM_CANDIDATES):
    # get candidates
    np.random.seed(1000)
    pred_means = np.random.rand(num_candidates)
    candidates = []
    for i in range(num_candidates):
        candidates.append(BernoulliRV(pred_means[i]))

    # get true maximum
    true_max = np.max(pred_means)
    true_max_indices = np.where(pred_means == true_max)

    # solve using uniform allocation
    obj = objectives.RandomBinaryObjective()
    ua = GittinsIndex98(obj, candidates)
    result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(MAX_ITERS * 10), snapshot_rate = SNAPSHOT_RATE)
    # NOTE: needs more iters on this problem

    # check result (not guaranteed to work in finite iterations but whatever)
    assert(len(result.best_candidates) == 1)
    assert(np.abs(result.best_candidates[0].p() - true_max) < 1e-4)
    logging.info('Gittins Indices test passed!')
    logging.info('Took %f sec' %(result.total_time))
    logging.info('Best index %d' %(true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Gittins Indices 98')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Gittins Indices 98')

    return result

def test_gaussian_uniform_alloc(num_candidates=NUM_CANDIDATES):
    # get candidates
    np.random.seed(1000)
    actual_means = np.random.rand(num_candidates)
    candidates = [BernoulliRV(m) for m in actual_means]

    # get true maximum
    true_max = np.max(actual_means)
    true_max_indices = np.where(actual_means == true_max)

    # solve using uniform allocation
    obj = objectives.RandomBinaryObjective()
    ua = GaussianUniformAllocationMean(obj, candidates)
    result = ua.solve(termination_condition=tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate=SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert len(result.best_candidates) == 1
    assert np.abs(result.best_candidates[0].p() - true_max) < 1e-4
    logging.info('Gaussian uniform allocation test passed!')
    logging.info('Took %f sec' % (result.total_time))
    logging.info('Best index %d' % (true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Gaussian Uniform Allocation')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Gaussian Uniform Allocation')

    return result

def test_gaussian_thompson_sampling(num_candidates=NUM_CANDIDATES):
    # get candidates
    np.random.seed(1000)
    actual_means = np.random.rand(num_candidates)
    candidates = [BernoulliRV(m) for m in actual_means]

    # get true maximum
    true_max = np.max(actual_means)
    true_max_indices = np.where(actual_means == true_max)

    # solve using Thompson sampling
    obj = objectives.RandomBinaryObjective()
    ts = GaussianThompsonSampling(obj, candidates)
    result = ts.solve(termination_condition=tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate=SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert len(result.best_candidates) == 1
    assert np.abs(result.best_candidates[0].p() - true_max) < 1e-4
    logging.info('Gaussian Thompson sampling test passed!')
    logging.info('Took %f sec' % (result.total_time))
    logging.info('Best index %d' % (true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Gaussian Thompson Sampling')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Gaussian Thompson Sampling')

    return result

def test_gaussian_ucb(num_candidates=NUM_CANDIDATES):
    # get candidates
    np.random.seed(1000)
    actual_means = np.random.rand(num_candidates)
    candidates = [BernoulliRV(m) for m in actual_means]

    # get true maximum
    true_max = np.max(actual_means)
    true_max_indices = np.where(actual_means == true_max)

    # solve using GP-UCB
    obj = objectives.RandomBinaryObjective()
    ts = GaussianUCBSampling(obj, candidates)
    result = ts.solve(termination_condition=tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate=SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert len(result.best_candidates) == 1
    assert np.abs(result.best_candidates[0].p() - true_max) < 1e-4
    logging.info('Gaussian UCB sampling test passed!')
    logging.info('Took %f sec' % (result.total_time))
    logging.info('Best index %d' % (true_max_indices[0]))

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Gaussian UCB Sampling')

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Gaussian UCB Sampling')

    return result

def test_correlated_thompson_sampling(num_candidates=NUM_CANDIDATES, sig=1.0, eps=0.5):
    # get candidates
    actual_means = np.linspace(0.0, 1.0, num=num_candidates)
    candidates = [BernoulliRV(m) for m in actual_means]

    # get true maximum
    true_max = np.max(actual_means)
    true_max_indices = np.where(actual_means == true_max)

    # constructing nearest neighbor and kernel
    def phi(bern):
        return np.array([round(bern.p(), 2)])
    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(sigma=sig, phi=phi)

    # solve using Thompson sampling
    obj = objectives.RandomBinaryObjective()
    ts = CorrelatedThompsonSampling(obj, candidates, nn, kernel, tolerance=eps)
    result = ts.solve(termination_condition=tc.MaxIterTerminationCondition(MAX_ITERS), snapshot_rate=SNAPSHOT_RATE)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert len(result.best_candidates) == 1
    assert np.abs(result.best_candidates[0].p() - true_max) < 1e-4
    logging.info('Correlated Thompson sampling test passed!')
    logging.info('Took %f sec' % (result.total_time))
    logging.info('Best index %d' % (true_max_indices[0]))

    info = u' (σ=%.1f, ɛ=%.3f)' %(sig, eps)

    # visualize result
    plot_num_pulls(result)
    plt.title('Observations Per Variable for Correlated Thompson Sampling' + info)

    plot_value_vs_time(result, candidates, true_max)
    plt.title('P(Success) versus Iterations for Correlated Thompson Sampling' + info)

    return result

def plot_gpucb_vs_thompson(num_candidates=100):
    global MAX_ITERS
    MAX_ITERS = 3000
    np.random.seed(1000)
    actual_means = np.random.rand(num_candidates)
    true_max = np.max(actual_means)
    candidates = [BernoulliRV(m) for m in actual_means]

    ts = test_thompson_sampling()
    gpucb = test_gaussian_ucb()
    gts_values = [candidates[m.best_pred_ind].value() for m in ts.models]
    gpucb_values = [candidates[m.best_pred_ind].value() for m in gpucb.models]

    plt.figure()
    plt.plot(ts.iters, gts_values, color='red', label='Thompson sampling', linewidth=2)
    plt.plot(gpucb.iters, gpucb_values, color='blue', label='GP-UCB', linewidth=2)
    plt.plot(ts.iters, true_max*np.ones(len(ts.iters)), color='green', label='True max', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')
    plt.legend(loc=4)

def test_ua_vs_thompson(num_trials = 20, num_candidates=1000, brute_iters=50000, max_iters=5000, snapshot_rate=20):
    # get candidates
    np.random.seed(1000)
    prior_dist = 'gaussian'

    # iterate through trials
    ua_results = []
    ts_results = []
    true_pfcs = []
    est_pfcs = []
    for i in range(num_trials):
        logging.info('Trial %d' %(i))

        # generate rangom candidates
        if prior_dist == 'gaussian':
            true_pfc = scipy.stats.norm.rvs(loc=0.5, scale=0.1, size=num_candidates) 
            true_pfc[true_pfc < 0] = 0
            true_pfc[true_pfc > 1] = 1
        else:
            true_pfc = np.random.rand(num_candidates)

        candidates = []
        for i in range(num_candidates):
            candidates.append(BernoulliRV(true_pfc[i]))

        # get true maximum
        true_max = np.max(true_pfc)
        true_max_indices = np.where(true_pfc == true_max)

        # solve using uniform allocation
        obj = objectives.RandomBinaryObjective()

        ua = UniformAllocationMean(obj, candidates)
        ua_result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(brute_iters), snapshot_rate = snapshot_rate)

        ts = ThompsonSampling(obj, candidates)
        ts_result = ts.solve(termination_condition = tc.MaxIterTerminationCondition(max_iters), snapshot_rate = snapshot_rate)

        # check result (not guaranteed to work in finite iterations but whatever)
        logging.info('UA took %f sec' %(ua_result.total_time))
        logging.info('UA best index %d' %(true_max_indices[0][0]))
        
        logging.info('TS took %f sec' %(ts_result.total_time))
        logging.info('TS best index %d' %(true_max_indices[0][0]))

        brute_model = ua_result.models[-1]

        true_pfcs.append(true_pfc)
        est_pfcs.append(models.BetaBernoulliModel.beta_mean(brute_model.alphas, brute_model.betas))
        ua_results.append(ua_result)
        ts_results.append(ts_result)

    # aggregate results wrt truth
    all_ua_norm_rewards = np.zeros([len(ua_results), len(ua_results[0].iters)]) 
    all_ts_norm_rewards = np.zeros([len(ts_results), len(ts_results[0].iters)])
    j = 0
    for true_pfc, result in zip(true_pfcs, ua_results):
        best_pfc = np.max(true_pfc)
        ua_pred_values = np.array([true_pfc[m.best_pred_ind] for m in result.models])
        all_ua_norm_rewards[j,:] = ua_pred_values / best_pfc
        j += 1

    j = 0
    for true_pfc, result in zip(true_pfcs, ts_results):
        best_pfc = np.max(true_pfc)
        ts_pred_values = np.array([true_pfc[m.best_pred_ind] for m in result.models])
        all_ts_norm_rewards[j,:] = ts_pred_values / best_pfc
        j += 1

    # aggregate results wrt est
    all_ua_norm_est_rewards = np.zeros([len(ua_results), len(ua_results[0].iters)]) 
    all_ts_norm_est_rewards = np.zeros([len(ts_results), len(ts_results[0].iters)])
    j = 0
    for est_pfc, result in zip(est_pfcs, ua_results):
        best_pfc = np.max(est_pfc)
        ua_pred_values = np.array([est_pfc[m.best_pred_ind] for m in result.models])
        all_ua_norm_est_rewards[j,:] = ua_pred_values / best_pfc
        j += 1

    j = 0
    for est_pfc, result in zip(est_pfcs, ts_results):
        best_pfc = np.max(est_pfc)
        ts_pred_values = np.array([est_pfc[m.best_pred_ind] for m in result.models])
        all_ts_norm_est_rewards[j,:] = ts_pred_values / best_pfc
        j += 1

    # params
    line_width = 2.5
    font_size = 15

    # histogram of all arms
    all_true_pfcs = np.zeros(0)
    for true_pfc in true_pfcs:
        all_true_pfcs = np.r_[all_true_pfcs, true_pfc]

    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins+1)
    plt.figure()
    n, bins, patches = plt.hist(all_true_pfcs, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)

    # visualize result
    ua_avg_norm_reward = np.mean(all_ua_norm_rewards, axis=0)
    ts_avg_norm_reward = np.mean(all_ts_norm_rewards, axis=0)

    plt.figure()
    plt.plot(ua_results[0].iters, ua_avg_norm_reward, c=u'b', linewidth=line_width, label='Uniform Allocation')
    plt.plot(ts_results[0].iters, ts_avg_norm_reward, c=u'g', linewidth=line_width, label='Thompson Sampling')

    plt.xlim(0, np.max(ts_results[0].iters))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    # visualize est result
    ua_avg_norm_est_reward = np.mean(all_ua_norm_est_rewards, axis=0)
    ts_avg_norm_est_reward = np.mean(all_ts_norm_est_rewards, axis=0)

    plt.figure()
    plt.plot(ua_results[0].iters, ua_avg_norm_est_reward, c=u'b', linewidth=line_width, label='Uniform Allocation')
    plt.plot(ts_results[0].iters, ts_avg_norm_est_reward, c=u'g', linewidth=line_width, label='Thompson Sampling')

    plt.xlim(0, np.max(ts_results[0].iters))
    plt.ylim(0.5, 1)
    plt.xlabel('Iteration', fontsize=font_size)
    plt.ylabel('Normalized Probability of Force Closure', fontsize=font_size)
    plt.title('Avg Estimated Normalized PFC vs Iteration', fontsize=font_size)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')

    plt.show()

    return result

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # test_uniform_alloc()
    # test_thompson_sampling()
    # test_gittins_indices_98()

    # test_gaussian_uniform_alloc()
    # test_gaussian_thompson_sampling()
    # test_gaussian_ucb()

    # test_correlated_thompson_sampling(eps=1e-3)
    # test_correlated_thompson_sampling(eps=0.3)
    # test_correlated_thompson_sampling(eps=0.7)
    # test_correlated_thompson_sampling(eps=0.98)

#    plot_gpucb_vs_thompson()
#    plt.show()
    test_ua_vs_thompson()
