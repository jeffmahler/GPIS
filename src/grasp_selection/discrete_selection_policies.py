"""
Policies for selecting the next point in discrete solvers

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import numpy as np
import IPython

import models

class DiscreteSelectionPolicy:
    __metaclass__ = ABCMeta

    def __init__(self, model):
        if not isinstance(model, models.DiscreteModel):
            raise ValueError('Must supply a discrete predictive model')
        self.model_ = model

    @abstractmethod
    def choose_next(self):
        """
        Choose the next index of the model to sample 
        """
        pass

class UniformSelectionPolicy(DiscreteSelectionPolicy):
    def choose_next(self):
        """ Returns an index uniformly at random"""
        num_vars = self.model_.num_vars()
        next_index = np.random.choice(num_vars)
        return next_index

class MaxDiscreteSelectionPolicy(DiscreteSelectionPolicy):
    def choose_next(self):
        """ Returns the index of the maximal variable, breaking ties uniformly at random"""
        max_indices, max_mean_vals, max_var_vals = self.model_.max_prediction()
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]

class ThompsonSelectionPolicy(DiscreteSelectionPolicy):
    """ Chooses the next point using the Thompson sampling selection policy"""
    def choose_next(self, stop = False):
        """ Returns the index of the maximal random sample, breaking ties uniformly at random"""
        sampled_values = self.model_.sample()
        if stop:
            IPython.embed()
        max_indices = np.where(sampled_values == np.max(sampled_values))[0]
        num_max_indices = max_indices.shape[0]
        next_index = np.random.choice(num_max_indices)
        return max_indices[next_index]        
