"""
Classes for predicting some (possibly non-deterministic) value over a set of discrete candidates or
continuous space

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import copy
import numpy as np
import scipy.stats

import IPython

class Model:
    """
    A predictor of some value of the input data
    """
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self.predict(x)

    @abstractmethod
    def predict(self, x):
        """
        Predict the function of the data at some point x. For probabilistic models this returns the mean prediction
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the model based on current data
        """
        pass

    @abstractmethod
    def snapshot(self):
        """
        Returns a concise description of the current model for debugging and logging purposes
        """
        pass

class DiscreteModel(Model):
    """
    Maintains a prediction over a discrete set of points
    """
    @abstractmethod
    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean predicted value
        """
        pass

    @abstractmethod
    def sample(self):
        """
        Sample discrete predictions from the model. For deterministic models, returns the deterministic prediction
        """
        pass

    def num_vars(self):
        """Returns the number of variables in the model"""
        return self.num_vars_


class Snapshot:
    __metaclass__ = ABCMeta
    def __init__(self, best_pred_ind, num_obs):
        self.best_pred_ind = best_pred_ind
        self.num_obs = copy.copy(num_obs)

class BernoulliSnapshot(Snapshot):
    def __init__(self, best_pred_ind, means, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.means = copy.copy(means)

class BetaBernoulliSnapshot(Snapshot):
    def __init__(self, best_pred_ind, alphas, betas, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.alphas = copy.copy(alphas)
        self.betas = copy.copy(betas)

class CorrelatedBetaBernoulliSnapshot(BetaBernoulliSnapshot):
    def __init__(self, best_pred_ind, alphas, betas, correlations, num_obs):
        BetaBernoulliSnapshot.__init__(self, best_pred_ind, alphas, betas, num_obs)
        self.correlations = copy.copy(correlations)

class GaussianSnapshot(Snapshot):
    def __init__(self, best_pred_ind, means, variances, num_obs):
        Snapshot.__init__(self, best_pred_ind, num_obs)
        self.means = copy.copy(means)
        self.variances = copy.copy(variances)

class BernoulliModel(DiscreteModel):
    """
    Standard bernoulli model for predictions over a discrete set of candidates
    Params:
        num_vars: (int) the number of variables to track
        prior_means: (float) prior on mean probabilty of success for candidates
    """
    def __init__(self, num_vars, mean_prior = 0.5):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to BetaBernoulliModel')

        self.num_vars_ = num_vars
        self.mean_prior_  = mean_prior

        self._init_model_params()

    def _init_model_params(self):
        """
        Allocates numpy arrays for the estimated alpha and beta values for each variable,
        and the number of observations for each
        """
        self.pred_means_ = self.mean_prior_ * np.ones(self.num_vars_)
        self.num_observations_ = np.zeros(self.num_vars_)

    @staticmethod
    def bernoulli_mean(p):
        """ Mean of the beta distribution with params alpha and beta """
        return p

    @staticmethod
    def bernoulli_variance(p, n):
        """ Uses Wald interval for variance prediction """
        sqrt_p_n = np.sqrt(p * (1 - p) / n)
        z = scipy.stats.norm.cdf(0.68) # cdf for standard Gaussian, 1 -sigma deviation
        return 2 * z * sqrt_p_n

    def predict(self, index):
        """
        Predicts the probability of success for the variable indexed by |index|
        """
        return BernoulliModel.bernoulli_mean(self.pred_means_[index])

    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean probaiblity of success
        """
        mean_posteriors = BernoulliModel.bernoulli_mean(self.pred_means_)
        max_indices = np.where(mean_posteriors == np.max(mean_posteriors))[0]
        max_posterior_means = mean_posteriors[max_indices]
        max_posterior_vars = BernoulliModel.bernoulli_variance(self.pred_means_[max_indices], self.num_observations_[max_indices])

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """
        Update the model based on an observation of |value| at index |index|
        """
        if value < 0 or value > 1:
            raise ValueError('Values must be between 0 and 1')

        self.pred_means_[index] = self.pred_means_[index] * (self.num_observations_[index] / (self.num_observations_[index] + 1)) + \
            value * (1.0 / (self.num_observations_[index] + 1));
        self.num_observations_[index] = self.num_observations_[index] + 1

    def snapshot(self):
        """
        Return copys of the model params
        """
        ind, mn, var = self.max_prediction()
        return BernoulliSnapshot(ind[0], self.pred_means_, self.num_observations_)

    def sample(self):
        """
        Samples probabilities of success from the given values
        """
        return self.pred_means_

class BetaBernoulliModel(DiscreteModel):
    """
    Beta-Bernoulli model for predictions over a discrete set of candidates
    Params:
        num_vars: (int) the number of variables to track
        alpha_prior and beta_prior: (float) the prior parameters of a Beta distribution over the
        probability of success for each candidate
    """
    def __init__(self, num_vars, alpha_prior = 1., beta_prior = 1.):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to BetaBernoulliModel')

        self.num_vars_ = num_vars
        self.alpha_prior_  = alpha_prior
        self.beta_prior_  = beta_prior

        self._init_model_params()

    def _init_model_params(self):
        """
        Allocates numpy arrays for the estimated alpha and beta values for each variable,
        and the number of observations for each
        """
        self.posterior_alphas_ = self.alpha_prior_ * np.ones(self.num_vars_)
        self.posterior_betas_ = self.beta_prior_ * np.ones(self.num_vars_)
        self.num_observations_ = np.zeros(self.num_vars_)

    @staticmethod
    def beta_mean(alpha, beta):
        """ Mean of the beta distribution with params alpha and beta """
        return alpha / (alpha + beta)

    @staticmethod
    def beta_variance(alpha, beta):
        """ Mean of the beta distribution with params alpha and beta """
        return (alpha * beta) / ( (alpha + beta)**2 * (alpha + beta + 1))

    @property
    def posterior_alphas(self):
        return self.posterior_alphas_

    @property
    def posterior_betas(self):
        return self.posterior_betas_

    def predict(self, index):
        """
        Predicts the probability of success for the variable indexed by |index|
        """
        return BetaBernoulliModel.beta_mean(self.posterior_alphas_[index], self.posterior_betas_[index])

    def max_prediction(self):
        """
        Returns the index (or indices), posterior mean, and posterior variance of the variable(s) with the
        maximal mean probaiblity of success
        """
        mean_posteriors = BetaBernoulliModel.beta_mean(self.posterior_alphas_, self.posterior_betas_)
        max_indices = np.where(mean_posteriors == np.max(mean_posteriors))[0]
        max_posterior_means = mean_posteriors[max_indices]
        max_posterior_vars = BetaBernoulliModel.beta_variance(self.posterior_alphas_[max_indices], self.posterior_betas_[max_indices])

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """
        Update the model based on an observation of |value| at index |index|
        """
        if value < 0 or value > 1:
            raise ValueError('Values must be between 0 and 1')

        self.posterior_alphas_[index] = self.posterior_alphas_[index] + value
        self.posterior_betas_[index] = self.posterior_betas_[index] + (1.0 - value)
        self.num_observations_[index] = self.num_observations_[index] + 1

    def snapshot(self):
        """
        Return copys of the model params
        """
        ind, mn, var = self.max_prediction()
        return BetaBernoulliSnapshot(ind[0], self.posterior_alphas_, self.posterior_betas_, self.num_observations_)

    def sample(self, vis = False, stop = False):
        """
        Samples probabilities of success from the given values
        """
        #samples = np.random.beta(self.posterior_alphas_, self.posterior_betas_)
        samples = scipy.stats.beta.rvs(self.posterior_alphas_, self.posterior_betas_)
        if stop:
            IPython.embed()
        if vis:
            print 'Samples'
            print samples
            print 'Estimated mean'
            print BetaBernoulliModel.beta_mean(self.posterior_alphas_, self.posterior_betas_)
            print 'At best index'
            print BetaBernoulliModel.beta_mean(self.posterior_alphas_[21], self.posterior_betas_[21])
        return samples

class GaussianModel(DiscreteModel):
    """Gaussian model for predictions over a discrete set of candidates.
    Params:
        num_vars: (int) the number of variables to track
        mean_prior: (float) prior parameter
        sigma: (float) noise
    """
    def __init__(self, num_vars, mean_prior=0.5, sigma=1e-2):
        if num_vars <= 0:
            raise ValueError('Must provide at least one variable to GaussianModel')

        self.num_vars_ = num_vars
        self.mean_prior_  = mean_prior
        self.sigma_ = sigma

        self._init_model_params()

    def _init_model_params(self):
        self.means_ = self.mean_prior_ * np.ones(self.num_vars_)
        self.vars_ = np.ones(self.num_vars_)
        self.num_observations_ = np.zeros(self.num_vars_)

    @property
    def means(self):
        return self.means_

    @property
    def variances(self):
        return self.vars_

    def predict(self, index):
        """Predict the value of the index'th variable.
        Params:
            index: (int) the variable to find the predicted value for
        """
        return self.means_[index]

    def max_prediction(self):
        """Returns the index, mean, and variance of the variable(s) with the
        maximal predicted value.
        """
        max_mean = np.max(self.means_)
        max_indices = np.where(self.means_ == max_mean)[0]
        max_posterior_means = self.means_[max_indices]
        max_posterior_vars = self.vars_[max_indices]

        return max_indices, max_posterior_means, max_posterior_vars

    def update(self, index, value):
        """Update the model based on current data.
        Params:
            index: (int) the index of the variable that was evaluated
            value: (float) the value of the variable
        """
        if not (0 <= value <= 1):
            raise ValueError('Values must be between 0 and 1')

        old_mean = self.means_[index]
        old_var = self.vars_[index]
        noise = self.sigma_ ** 2

        self.means_[index] = old_mean + ((value - old_mean) * old_var) / (old_var + noise)
        self.vars_[index] = old_var - (old_var ** 2) / (old_var + noise)
        self.num_observations_[index] += 1

    def sample(self, stop=False):
        """Sample discrete predictions from the model."""
        samples = scipy.stats.multivariate_normal.rvs(self.means_, self.vars_)
        if stop:
            IPython.embed()
        return samples

    def snapshot(self):
        """Returns a concise description of the current model for debugging and
        logging purposes.
        """
        ind, mn, var = self.max_prediction()
        return GaussianSnapshot(ind[0], self.means_, self.vars_, self.num_observations_)

class CorrelatedBetaBernoulliModel(BetaBernoulliModel):
    """Correlated Beta-Bernoulli model for predictions over a discrete set of
    candidates.
    Params:
        candidates: the objects to track
        nn: a NearestNeighbor instance to use for neighborhood lookups
        kernel: a Kernel instance to measure similarities
        tolerance: (float) for computing radius of neighborhood, between 0 and 1
        alpha_prior and beta_prior: (float) the prior parameters of a Beta
        distribution over the probability of success for each candidate
    """
    def __init__(self, candidates, nn, kernel, tolerance=1e-2,
                 alpha_prior=1.0, beta_prior=1.0):
        BetaBernoulliModel.__init__(self, len(candidates), alpha_prior, beta_prior)
        self.candidates_ = candidates

        self.kernel_ = kernel
        self.tolerance_ = tolerance
        self.error_radius_ = kernel.error_radius(tolerance)
        self.kernel_matrix_ = None

        self.nn_ = nn
        self.nn_.train(candidates)

    def kernel_matrix(self):
        """
        Create the full kernel matrix for debugging purposes
        """
        if self.kernel_matrix_ is None:
            self.kernel_matrix_ = np.zeros([self.num_vars_, self.num_vars_])
            i = 0
            for candidate_i in candidates:
                j = 0
                for candidate_j in candidates:
                    self.kernel_matrix_[i,j] = self.kernel_(candidate_i, candidate_j)
                    j += 1
                i += 1
        return self.kernel_matrix_

    def update(self, index, value):
        """Update the model based on current data
        Params:
            index: (int) the index of the variable that was evaluated
            value: (float) the value of the variable
        """
        if not (0 <= value <= 1):
            raise ValueError('Values must be between 0 and 1')

        # find neighbors within radius
        candidate = self.candidates_[index]
        neighbor_indices, _ = self.nn_.within_distance(candidate, self.error_radius_,
                                                       return_indices=True)

        # create array of correlations
        correlations = np.zeros(self.num_vars_)
        for neighbor_index in neighbor_indices:
            neighbor = self.candidates_[neighbor_index]
            correlations[neighbor_index] = self.kernel_(candidate, neighbor)

        self.posterior_alphas_ = self.posterior_alphas_ + value * correlations
        self.posterior_betas_ = self.posterior_betas_ + (1.0 - value) * correlations
        # TODO: should num_observations_ be updated by correlations instead?
        self.num_observations_[index] += 1

    def snapshot(self):
        """
        Return copys of the model params
        """
        ind, mn, var = self.max_prediction()
        return CorrelatedBetaBernoulliSnapshot(ind[0], self.posterior_alphas_, self.posterior_betas_, self.kernel_matrix_, self.num_observations_)
