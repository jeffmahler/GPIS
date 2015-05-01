from abc import ABCMeta, abstractmethod

class Solver:
    __metaclass__ = ABCMeta

    def __init__(self, objective):
        self.objective_ = objective

    def solve(self):
        '''
        Solves for the maximal / minimal point
        '''
        return self.top_K_points(1)

    @abstractmethod
    def top_K_solve(self, K):
        '''
        Solves for the top K maximal / minimal points
        '''
        pass

class DiscreteSamplingSolver(Solver):
    __metaclass__ = ABCMeta
    def __init__(self, objective, candidates):
        """
        Initialize a solver with a discrete set of candidate points
        specified in a list object
        """
        self.candidates_ = candidates # discrete candidates
        self.num_candidates_ = len(candidates)
        Solver.__init__(self, objective)

    @abstractmethod
    def discrete_maximize(self, candidates):
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

    def top_K_solve(self, K):
        '''
        Solves for the top K maximal / minimal points
        '''
        # partition the input space
        candidate_bins = self.partition(K)

        # maximize over each bin
        top_K_results = []
        for k in range(K):
            top_K_results = discrete_maximize(candidate_bins[k])
        return top_K_results

class AdaptiveSamplingResult:
    """
    Basically a struct to store the results of sampling / optimization
    """
    def __init__(self, best_candidate, indices, vals, models):
        self.best_candidate = best_candidate
        self.indices = indices
        self.vals = vals
        self.models = models

class DiscreteAdaptiveSampler(DiscreteSamplingSolver):
    
    def discrete_maximize(self, candidates, termination_condition, snapshot_rate = 1):
        """
        Maximizes a function over a discrete set of variables by
        iteratively predicting the best point (using some model policy) 
        """
        # check input
        if len(candidate) == 0:
            raise ValueError('No candidates specified')

        if not isinstance(self.model_, models.DiscretePredictiveModel):
            logging.error('Illegal model specified')
            raise ValueError('Illegitimate model used in DiscreteAdaptiveSampler')

        # init vars
        terminate = False
        k = 0 # cur iter
        self.model_.reset(candidates) # update model with new candidates

        # logging
        iter_indices = []
        iter_vals = []
        iter_models = []
        
        while not terminate:
            # get next point to sample    
            next_ind = self.model_.choose_next()

            # evaluate the function at the given point (can be nondeterministic)
            next_ind_val = self.objective_.evaluate(candidates[next_ind])

            # snapshot the model and whatnot
            if (k % snapshot_rate) == 0:
                iter_indices.push_back(next_ind)
                iter_vals.push_back(next_ind_val)
                iter_models.push_back(self.model_.snapshot())

            # update the model (e.g. posterior update, grasp pruning)
            self.model_.update(next_ind, next_ind_val)

            # check termination condiation
            terminate = termination_condition(k, self.objective_, self.model_)

        # log results and return
        best_ind = self.model_.best()
        best_candidate = candidates[best_ind]
        return AdaptiveSamplingResult(best_candidate, iter_indices, iter_vals, iter_models)        

# TODO: make this work someday...
class OptimizationSolver(Solver):
    pass
