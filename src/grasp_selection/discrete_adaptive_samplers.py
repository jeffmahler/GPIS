"""
Classes for selecting a candidate that maximizes some objective over a discrete set of candidates

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import discrete_selection_policies as dcsp
import models
import objectives
import solvers
import termination_conditions as tc

import IPython

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
        iters = []
        iter_indices = []
        iter_vals = []
        iter_models = []
        
        while not terminate:
            # get next point to sample    
            next_ind = self.selection_policy_.choose_next()

            # evaluate the function at the given point (can be nondeterministic)
            next_ind_val = self.objective_.evaluate(candidates[next_ind])

            # snapshot the model and whatnot
            if (k % snapshot_rate) == 0:
#                self.selection_policy_.choose_next(stop=True)
#                s = self.model_.sample(vis=True, stop=True)
#                IPython.embed()
                logging.info('Iteration %d' %(k))
                iters.append(k)
                iter_indices.append(next_ind)
                iter_vals.append(next_ind_val)
                iter_models.append(self.model_.snapshot())

            # update the model (e.g. posterior update, grasp pruning)
            self.model_.update(next_ind, next_ind_val)

            # check termination condiation
            terminate = termination_condition(k, self.objective_, self.model_)
            
            k = k + 1

        # log results and return
        best_indices, best_pred_means, best_pred_vars = self.model_.max_prediction()
        best_candidates = []
        num_best = best_indices.shape[0]
        for i in range(num_best):
            best_candidates.append(candidates[best_indices[i]])
        return AdaptiveSamplingResult(best_candidates, best_pred_means, best_pred_vars,
                                      iters, iter_indices, iter_vals, iter_models)        


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
        self.selection_policy_.set_model(self.model_)
        # always update the selection policy!    

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

class AdaptiveSamplingResult:
    """
    Basically a struct to store the results of sampling / optimization
    """
    def __init__(self, best_candidates, best_pred_means, best_pred_vars, iters, indices, vals, models):
        self.best_candidates = best_candidates
        self.best_pred_means = best_pred_means
        self.best_pred_vars = best_pred_vars
        self.iters = iters
        self.indices = indices
        self.vals = vals
        self.models = models

class BernoulliRV:
    def __init__(self, p):
        self.p_ = p

    def sample_success(self):
        return scipy.stats.bernoulli.rvs(self.p_)

    def p(self):
        return self.p_


def test_uniform_alloc(num_candidates = 100):
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

    result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(10000), snapshot_rate = 1000)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert(len(result.best_candidates) == 1)
    assert(np.abs(result.best_candidates[0].p() - true_max) < 1e-4)
    logging.info('Uniform alloc test passed!')
    logging.info('Best index %d' %(true_max_indices[0]))

    # visualize result
    ind = np.arange(num_candidates)
    width = 1

    fig, ax = plt.subplots()
    ax.bar(ind+width, result.models[-1].num_obs)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Num observations')
    plt.title('Observations Per Variable for Uniform Allocation')

    best_values = [candidates[m.best_pred_ind].p() for m in result.models]
    plt.figure()
    plt.plot(result.iters, best_values, color='blue', linewidth=2)
    plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')
    plt.title('P(Success) versus Iterations for Uniform Allocation')

def test_thompson_sampling(num_candidates = 100):
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
    ua = ThompsonSampling(obj, candidates)

    result = ua.solve(termination_condition = tc.MaxIterTerminationCondition(10000), snapshot_rate = 1000)

    # check result (not guaranteed to work in finite iterations but whatever)
    assert(len(result.best_candidates) == 1)
    assert(np.abs(result.best_candidates[0].p() - true_max) < 1e-4)
    logging.info('Thompson sampling test passed!')
    logging.info('Best index %d' %(true_max_indices[0]))

    # visualize result
    ind = np.arange(num_candidates)
    width = 1

    fig, ax = plt.subplots()
    ax.bar(ind+width, result.models[-1].num_obs)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Num observations')
    plt.title('Observations Per Variable for Thompson Sampling')

    best_values = [candidates[m.best_pred_ind].p() for m in result.models]
    plt.figure()
    plt.plot(result.iters, best_values, color='blue', linewidth=2)
    plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')
    plt.title('P(Success) versus Iterations for Thompson Sampling')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_uniform_alloc()
    test_thompson_sampling()
    plt.show()
