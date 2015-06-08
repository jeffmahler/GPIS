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
        self.predict(x)
    
    @abstractmethod
    def predict(self, x):
        """
        Predict the a function of the data at some point x. For probabilistic models this returns the mean prediction
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

class BernoulliSnapshot:
    def __init__(self, best_pred_ind, means, num_obs):
        self.best_pred_ind = best_pred_ind
        self.means = copy.copy(means)
        self.num_obs = copy.copy(num_obs)

class BetaBernoulliSnapshot:
    def __init__(self, best_pred_ind, alphas, betas, num_obs):
        self.best_pred_ind = best_pred_ind
        self.alphas = copy.copy(alphas)
        self.betas = copy.copy(betas)
        self.num_obs = copy.copy(num_obs)

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
