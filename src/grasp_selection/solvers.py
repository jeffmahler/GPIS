"""
Abstract classes for solvers

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import termination_conditions as tc
import IPython

DEF_MAX_ITER = 100

class Solver:
    __metaclass__ = ABCMeta

    def __init__(self, objective):
        self.objective_ = objective

    @abstractmethod
    def solve(self, termination_condition = tc.MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        '''
        Solves for the maximal / minimal point
        '''
        pass


class TopKSolver(Solver):
    def __init__(self, objective):
        Solver.__init__(self, objective)

    @abstractmethod
    def top_K_solve(self, K, termination_condition = tc.MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        '''
        Solves for the top K maximal / minimal points
        '''
        pass

class SamplingSolver(Solver):
    """ Optimization methods based on a sampling strategy"""
    __metaclass__ = ABCMeta

class DiscreteSamplingSolver(TopKSolver):
    __metaclass__ = ABCMeta
    def __init__(self, objective, candidates):
        """
        Initialize a solver with a discrete set of candidate points
        specified in a list object
        """
        self.candidates_ = candidates # discrete candidates
        self.num_candidates_ = len(candidates)
        TopKSolver.__init__(self, objective)

    @abstractmethod
    def discrete_maximize(self, candidates, termination_condition, snapshot_rate):
        """
        Main loop for sampling-based solvers
        """
        pass

    def partition(self, K):
        """
        Partition the input space into K bins uniformly at random
        """
        candidate_bins = []
        indices = np.linspace(0, self.num_candidates_)
        indices_shuff = np.random.shuffle(indices) 
        candidates_per_bin = math.floor(float(self.num_candidates_) / float(K))

        # loop through bins, adding candidates at random
        start_i = 0
        end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
        for k in range(K-1):
            candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])

            start_i = start_i + candidates_per_bin
            end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
            
        candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])
        return candidate_bins

    def solve(self, termination_condition = tc.MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        """ Call discrete maxmization function with all candidates """
        return self.discrete_maximize(self.candidates_, termination_condition, snapshot_rate)

    def top_K_solve(self, K, termination_condition = tc.MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        """ Solves for the top K maximal / minimal points """
        # partition the input space
        if K == 1:
            candidate_bins = [self.candidates_]
        else:
            candidate_bins = self.partition(K)

        # maximize over each bin
        top_K_results = []
        for k in range(K):
            top_K_results.append(self.discrete_maximize(candidate_bins[k], termination_condition, snapshot_rate))
        return top_K_results

# TODO: make this work someday...
class OptimizationSolver(Solver):
    pass
