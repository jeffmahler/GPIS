from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import numpy as np
import time

import objectives
import solvers
import termination_conditions as tc

import IPython

class IterativeLocalOptimizationResult:
    """
    Basically a struct to store the results of sampling / optimization
    """
    def __init__(self, best_x, best_f, best_grad_f, best_hess_f, total_time, checkpt_times, iters, vals_x, vals_f, grads_f, hess_f):
        self.best_x = best_x
        self.best_f = best_f
        self.best_grad_f = best_grad_f
        self.best_hess_f = best_hess_f
        self.total_time = total_time
        self.checkpt_times = checkpt_times
        self.iters = iters
        self.vals_x = vals_x
        self.vals_f = vals_f
        self.grads_f = grads_f
        self.hess_f = hess_f

class IterativeLocalOptimizer(solvers.OptimizationSolver):
    __metaclass__ = ABCMeta

    def __init__(self, objective):
        solvers.OptimizationSolver.__init__(self, objective)

    @abstractmethod
    def update(self, x):
        """
        Computes the next point based on the current x
        Returns:
            next_x, next_f, next_f_grad, next_f_hess
            the latter two can be None if not computed in the update
        """
        pass

    def solve(self, termination_condition, snapshot_rate, start_x, true_x):
        """
        Optimizes an objective iteratively by calling the 'next point' function
        """
        if not self.is_feasible(start_x):
            logging.warning('Starting x value is not feasible for the problem')
            return None

        # init vars
        terminate = False
        k = 0
        cur_x = start_x

        # logging
        times = []
        iters = []
        iter_x = []
        iter_f = []
        iter_grads = []
        iter_hess = []
        checkpt_times = []
        start_time = time.clock()
        cur_f = self.objective_(cur_x)

        while not terminate:
            # update to next point and evaluate
            prev_f = cur_f
            cur_x, cur_f, cur_grad_f, cur_hess_f = self.update(cur_x, k)

            # correct for minimization being used as a maximization
            """
            if isinstance(self.objective_, objectives.MinimizationObjective):
                cur_f = -cur_f
                if cur_grad_f is not None:
                    cur_grad_f = -cur_grad_f
                if cur_hess_f is not None:
                    cur_hess_f = -cur_hess_f
                    """
            # snapshot the model and whatnot
            if (k % snapshot_rate) == 0:
                logging.info('Iteration %d' %(k))

                # log time and stuff
                checkpt = time.clock()
                times.append(checkpt - start_time)
                iters.append(k)
                iter_x.append(cur_x)

                iter_f.append(cur_f)
                if cur_grad_f is not None:
                    iter_grads.append(cur_grad_f)
                if cur_hess_f is not None:
                    iter_hess.append(cur_hess_f)

            # check termination condiation
            terminate = termination_condition(k, cur_f, prev_f, cur_grad = cur_grad_f, cur_hess = cur_hess_f)
            k = k + 1

        # log total runtime
        end_time = time.clock()
        total_duration = end_time - start_time

        # log results and return
        best_x = cur_x
        best_f = cur_f
        best_grad_f = cur_grad_f
        best_hess_f = cur_hess_f
        return IterativeLocalOptimizationResult(best_x, best_f, best_grad_f, best_hess_f, total_duration, checkpt_times,
                                                iters, iter_x, iter_f, iter_grads, iter_hess)

class StepPolicy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step_size(self, objective, x, iteration):
        """
        Compute the next step size
        """
        pass

class BasicStepPolicy(StepPolicy):
    def __init__(self, step_size=1e-1):
        self.size = step_size

    def step_size(self, objective, x, iteration):
        return self.size

class DecayingStepPolicy(BasicStepPolicy):
    def __init__(self, step_size=1e-1, decay_rate=3):
        self.size = step_size
        self.decay_rate = decay_rate

    def step_size(self, objective, x, iteration):
        return self.size / float(iteration//self.decay_rate + 1)

class BacktrackingLSPolicy(StepPolicy):
    def __init__(self, alpha = 0.45, beta = 0.9):
        self.alpha_ = alpha
        self.beta_ = beta

    def step_size(self, objective, x, iteration, f = None, grad_f = None, hess_f = None):
        # only query objective if extra data not provided
        if f is None:
            f = objective(x)
        if grad_f is None:
            grad_f = objective.gradient(x)

        # standard BTLS algorithm
        eta = 1
        while objective(x + eta * grad_f) < f + self.alpha_ * eta * grad_f.T.dot(grad_f):
            eta = self.beta_ * eta
        return eta

class UnconstrainedGradientAscent(IterativeLocalOptimizer):
    def __init__(self, objective, step_policy):
        if not isinstance(objective, objectives.DifferentiableObjective):
            raise ValueError('Illegal objective specified')

        # for now only BTLS supported
        if not isinstance(step_policy, (BacktrackingLSPolicy, BasicStepPolicy)):
            raise ValueError('Illegal step size policy specified')

        self.step_policy_ = step_policy
        IterativeLocalOptimizer.__init__(self, objective)

    def update(self, x, iteration, f = None, grad_f = None, hess_f = None):
        """ Compute the next point based on gradient ascent """
        if f is None:
            f = self.objective_(x)
        if grad_f is None:
            grad_f = self.objective_.gradient(x)

        # get step size and update gradient
        step_size = self.step_policy_.step_size(self.objective_, x, iteration)
        next_x = x + step_size * grad_f

        # function value at new point
        next_f = self.objective_(next_x)
        next_grad_f = self.objective_.gradient(next_x)
        return next_x, next_f, next_grad_f, None

def plot_value_vs_time_gradient(result, true_max, title=''):
    plt.figure()
    plt.plot(result.iters, result.vals_f, color='blue', linewidth=2)
    plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title(title)

def test_grad_ascent():
    np.random.seed(100)

    # init vars
    x_dim = int(10)
    b_dim = int(5)
    A = np.random.rand(b_dim, x_dim)
    b = np.random.rand(b_dim)
    x_0 = np.random.rand(x_dim)
    objective = objectives.MinimizationObjective(objectives.LeastSquaresObjective(A, b))

    # get actual solution
    try:
        true_best_x = np.linalg.solve(A.T.dot(A), A.T.dot(b))
    except np.linalg.LinAlgError:
        logging.error('A transpose A ws not invertible!')
    true_best_f = objective(true_best_x)

    # run gradient ascent
    step_policy = BacktrackingLSPolicy()
    optimizer = UnconstrainedGradientAscent(objective, step_policy)
    result = optimizer.solve(termination_condition = tc.MaxIterTerminationCondition(100), snapshot_rate = 1, start_x = x_0, true_x = true_best_x)

    assert(np.abs(np.linalg.norm(result.best_f - true_best_f)) < 1e-2)

    logging.info('Val at true best x: %f' %(true_best_f))
    logging.info('Val at estimated best x: %f' %(result.best_f))
    plot_value_vs_time_gradient(result, true_best_f, 'Least squares grad ascent')

def test_grad_ascent_logistic():
    np.random.seed(100)

    num_from_zero = 100
    zero_mean, zero_cov = np.random.rand(2), 3e-2 * np.diag(np.random.rand(2))
    num_from_one = 100
    one_mean, one_cov = np.random.rand(2), 3e-2 * np.diag(np.random.rand(2))
    from_zero = np.random.multivariate_normal(zero_mean, zero_cov, num_from_zero)
    from_one = np.random.multivariate_normal(one_mean, one_cov, num_from_one)

    X = np.hstack([np.ones((num_from_zero + num_from_one, 1)),
                   np.vstack([from_zero, from_one])])
    y = np.hstack([np.zeros(num_from_zero), np.ones(num_from_one)])

    objective = objectives.MinimizationObjective(objectives.LogisticCrossEntropyObjective(X, y))

    # get ground truth solution from scikit-learn
    from sklearn.linear_model import LogisticRegression
    solver = LogisticRegression(fit_intercept=False)
    solver.fit(X, y)
    solution = solver.coef_.squeeze()
    solver_score = sum(solver.predict(X) == y)
    true_best_f = objective(solution)

    logging.info('scikit-learn solution: %s', solution)
    logging.info('scikit-learn score: %f', true_best_f)
    logging.info('scikit-learn iterations: %d', solver.n_iter_)

    # run gradient ascent
    step_policy = DecayingStepPolicy(1)
    optimizer = UnconstrainedGradientAscent(objective, step_policy)
    result = optimizer.solve(termination_condition=tc.MaxIterTerminationCondition(100),
                             snapshot_rate=1, start_x=np.random.rand(3), true_x=solution)
    logging.info('our solution: %s', result.best_x)
    logging.info('our score: %f', result.best_f)

    logging.info('Val at true best x: %f' %(true_best_f))
    logging.info('Val at estimated best x: %f' %(result.best_f))
    plot_value_vs_time_gradient(result, true_best_f, 'Logistic GA')


def test_stochastic_grad_ascent_logistic():
    np.random.seed(100)

    num_from_zero = 100
    zero_mean, zero_cov = np.random.rand(2), 3e-2 * np.diag(np.random.rand(2))
    num_from_one = 100
    one_mean, one_cov = np.random.rand(2), 3e-2 * np.diag(np.random.rand(2))
    from_zero = np.random.multivariate_normal(zero_mean, zero_cov, num_from_zero)
    from_one = np.random.multivariate_normal(one_mean, one_cov, num_from_one)

    X = np.hstack([np.ones((num_from_zero + num_from_one, 1)),
                   np.vstack([from_zero, from_one])])
    y = np.hstack([np.zeros(num_from_zero), np.ones(num_from_one)])

    objective = objectives.MinimizationObjective(
        objectives.StochasticLogisticCrossEntropyObjective(X, y, batch_size=50))

    # get ground truth solution from scikit-learn
    from sklearn.linear_model import LogisticRegression
    solver = LogisticRegression(fit_intercept=False)
    solver.fit(X, y)
    solution = solver.coef_.squeeze()
    solver_score = sum(solver.predict(X) == y)
    true_best_f = objective(solution)

    logging.info('scikit-learn solution: %s', solution)
    logging.info('scikit-learn score: %f', true_best_f)
    logging.info('scikit-learn iterations: %d', solver.n_iter_)

    # run gradient ascent
    step_policy = DecayingStepPolicy(1)
    optimizer = UnconstrainedGradientAscent(objective, step_policy)
    result = optimizer.solve(termination_condition=tc.MaxIterTerminationCondition(100),
                             snapshot_rate=1, start_x=np.random.rand(3), true_x=solution)
    logging.info('our solution: %s', result.best_x)
    logging.info('our score: %f', result.best_f)

    logging.info('Val at true best x: %f' %(true_best_f))
    logging.info('Val at estimated best x: %f' %(result.best_f))
    plot_value_vs_time_gradient(result, true_best_f, 'Logistic SGA')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_grad_ascent()
    test_grad_ascent_logistic()
    test_stochastic_grad_ascent_logistic()
    plt.show()
