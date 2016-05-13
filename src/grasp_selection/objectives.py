"""
Objectives that place some value on a set on input points

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numbers
import numpy as np
import scipy.stats as ss

import kernels

class Objective:
    __metaclass__ = ABCMeta

    def __call__(self, x):
        """ Evaluate the objective at a point x """
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x):
        """ Evaluates a function to be maximized at some point x """
        pass

    @abstractmethod
    def check_valid_input(self):
        """ Return whether or not a point is valid for the objective """
        pass

class DifferentiableObjective(Objective):
    __metaclass__ = ABCMeta

    @abstractmethod
    def gradient(self, x):
        """ Evaluate the gradient at x """
        pass

    @abstractmethod
    def hessian(self, x):
        """ Evaluate the hessian at x """
        pass

class MaximizationObjective(DifferentiableObjective):
    """
    Maximization on some supplied objective function. Actually not super important, here for symmetry
    """
    def __init__(self, obj):
        """ obj is the objective to call """
        if not isinstance(obj, Objective):
            raise ValueError("Function must be a single argument objective")
        self.obj_ = obj

    def check_valid_input(self, x):
        self.obj_.check_valid_input(x)

    def evaluate(self, x):
        return self.obj_(x)

    def gradient(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return self.obj_.gradient(x)

    def hessian(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return self.obj_.hessian(x)

class MinimizationObjective(DifferentiableObjective):
    """
    Maximization on some supplied objective function. Actually not super important, here for symmetry
    """
    def __init__(self, obj):
        """ obj is the objective to call """
        if not isinstance(obj, Objective):
            raise ValueError("Function must be a single argument objective")
        self.obj_ = obj

    def check_valid_input(self, x):
        self.obj_.check_valid_input(x)

    def evaluate(self, x):
        """ Return negative, as all solvers will be assuming a maximization """
        return -self.obj_(x)

    def gradient(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return -self.obj_.gradient(x)

    def hessian(self, x):
        if not isinstance(self.obj_, DifferentiableObjective):
            raise ValueError("Objective is non-differentiable")
        return -self.obj_.hessian(x)

class NonDeterministicObjective(Objective):

    def __init__(self, det_objective):
        """ Wraps around a deterministic objective """
        self.det_objective_ = det_objective

    def evaluate(self, x):
        """ Sample the input space, then evaluate """
        if not hasattr(x, "sample"):
            raise ValueError("Data points must have a sampling function returning a 0 or 1")

        x_val = x.sample()
        return self.det_objective_.evaluate(x_val)

class ZeroOneObjective(Objective):
    """ Zero One Loss based on thresholding """
    def __init__(self, b = 0):
        self.b_ = b

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Zero-One objective can only be evaluated on numbers")

    def evaluate(self, x):
        self.check_valid_input(x)
        return x >= self.b_

class IdentityObjective(Objective):
    """ Zero One Loss based on thresholding """
    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Zero-One objective can only be evaluated on numbers")

    def evaluate(self, x):
        self.check_valid_input(x)
        return x

class RandomBinaryObjective(NonDeterministicObjective):
    """
    Returns a 0 or 1 based on some underlying random probability of success for the data points
    Evaluated data points must have a sample_success method that returns 0 or 1
    """
    def __init__(self):
        NonDeterministicObjective.__init__(self, ZeroOneObjective(0.5))

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Random binary objective can only be evaluated on numbers")

class RandomContinuousObjective(NonDeterministicObjective):
    """
    Returns a continuous value based on some underlying random probability of success for the data points
    Evaluated data points must have a sample method
    """
    def __init__(self):
        NonDeterministicObjective.__init__(self, IdentityObjective())

    def check_valid_input(self, x):
        """ Check whether or not input is valid for the objective """
        if not isinstance(x, numbers.Number):
            raise ValueError("Random continuous objective can only be evaluated on numbers")

class LeastSquaresObjective(DifferentiableObjective):
    """ Classic least-squares loss 0.5 * |Ax - b|**2 """
    def __init__(self, A, b):
        self.A_ = A
        self.b_ = b

        self.x_dim_ = A.shape[1]
        self.b_dim_ = A.shape[0]
        if self.b_dim_ != b.shape[0]:
            raise ValueError('A and b must have same dimensions')

    def check_valid_input(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError('Least squares objective only works with numpy ndarrays!')
        if x.shape[0] != self.x_dim_:
            raise ValueError('x values must have same dimensions as number of columns of A')

    def evaluate(self, x):
        self.check_valid_input(x)
        return 0.5 * (x.T.dot(self.A_.T).dot(self.A_).dot(x) - 2 * self.b_.T.dot(self.A_).dot(x) + self.b_.T.dot(self.b_))

    def gradient(self, x):
        self.check_valid_input(x)
        return self.A_.T.dot(self.A_).dot(x) - self.A_.T.dot(self.b_)

    def hessian(self, x):
        self.check_valid_input(x)
        return self.A_.T.dot(self.A_)

class LogisticCrossEntropyObjective(DifferentiableObjective):
    def __init__(self, X, y):
        self.X_ = X
        self.y_ = y

    def check_valid_input(self, beta):
        if not isinstance(beta, np.ndarray):
            raise ValueError('Logistic cross-entropy objective only works with np.ndarrays!')
        if self.X_.shape[1] != beta.shape[0]:
            raise ValueError('beta dimension mismatch')

    def _mu(self, X, beta):
        return 1.0 / (1.0 + np.exp(-np.dot(X, beta)))

    def evaluate(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return -np.sum(self.y_ * np.log(mu) + (1 - self.y_) * np.log(1 - mu))

    def gradient(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return 2 * beta - np.dot(self.X_.T, self.y_ - mu)

    def hessian(self, beta):
        self.check_valid_input(beta)
        mu = self._mu(self.X_, beta)
        return 2 - np.dot(np.dot(self.X_.T, np.diag(mu * (1 - mu))), self.X_)

class StochasticLogisticCrossEntropyObjective(LogisticCrossEntropyObjective):
    def __init__(self, X, y, batch_size=1):
        LogisticCrossEntropyObjective.__init__(self, X, y)
        self.batch_size = batch_size

    def get_random_datum(self):
        num_data = self.y_.shape[0]
        indices = range(num_data) * (self.batch_size // num_data)
        if self.batch_size % num_data != 0:
            indices += list(np.random.randint(num_data, size=self.batch_size % num_data))
        x = self.X_[indices, :]
        y = self.y_[indices]
        return x, y

    def gradient(self, beta):
        self.check_valid_input(beta)
        x, y = self.get_random_datum()
        mu = self._mu(x, beta)
        return (2 * beta - np.dot(x.T, y - mu)).squeeze()

    def hessian(self, beta):
        self.check_valid_input(beta)
        x, y = self.get_random_datum()
        mu = self._mu(x, beta)
        return 2 - np.dot(np.dot(x.T, np.diag(mu * (1 - mu))), x)

class StochasticGraspWeightObjective(DifferentiableObjective):
    def __init__(self, X, S, F, config):
        self.X_ = X     # design matrix
        self.S_ = S     # num successes
        self.F_ = F     # num failures
        self.N_ = S + F # num trials

        self.mu_ = S / (S + F)
        self.batch_size = 1 # hard-coded for now
        self.num_grasps_ = S.shape[0]
        self.num_features_ = X.shape[1]

        self.config_ = config

        if not (X.shape[0] == S.shape[0] == F.shape[0]):
            raise ValueError('Dimension mismatch: %d %d %d' %(X.shape[0], S.shape[0], F.shape[0]))

    def check_valid_input(self, w):
        if not isinstance(w, np.ndarray):
            raise ValueError('Grasp weight objective only works with np.ndarrays')
        if w.shape[0] != self.num_features_:
            raise ValueError('weight vector dimension mismatch: %d %d' %(w.shape[0], self.num_features_))

    def kernel(self, w):
        def phi(row):
            return w * row
        return kernels.SquaredExponentialKernel(
            sigma=self.config_['kernel_sigma'], l=self.config_['kernel_l'], phi=phi)

    def evaluate(self, w):
        self.check_valid_input(w)
        kernel = self.kernel(w)

        # vectorized equivalent(?) not good for approximations
        # kernel_matrix = kernel.matrix(self.X_)
        # alpha = 1 + np.dot(kernel_matrix, self.S_) - self.S_
        # beta = 1 + np.dot(kernel_matrix, self.F_) - self.F_
        # return -np.sum(self.mu_ * np.log(alpha) + \
        #                (1 - self.mu_) * np.log(beta) - \
        #                np.log(alpha + beta))

        total = 0
        for i in range(self.num_grasps_):
            xi = self.X_[i, :]
            kernels = [kernel(xi, xj) for xj in self.X_]
            alphas = kernels * self.S_
            betas = kernels * self.F_
            alpha_i = 1 + np.sum(alphas) - alphas[i]
            beta_i = 1 + np.sum(betas) - betas[i]
            total += self.mu_[i] * np.log(alpha_i) + \
                     (1 - self.mu_[i]) * np.log(beta_i) - \
                     np.log(alpha_i + beta_i)
        return -total

    def get_random_datum(self):
        i = np.random.randint(self.num_grasps_, size=self.batch_size)
        x = self.X_[i, :]
        without_x = np.delete(self.X_, i, 0)
        return x, without_x, i

    def gradient(self, w):
        self.check_valid_input(w)
        kernel = self.kernel(w)
        x, without_x, i = self.get_random_datum()


        num_indices = self.config_['partial_gradient_size']
        if num_indices in (None, 0, self.num_grasps_):
            X = np.delete(self.X_, i, axis=0)
            S, F = np.delete(self.S_, i), np.delete(self.F_, i)
        elif num_indices > self.num_grasps_:
            raise ValueError('partial_gradient_size is too large.')
        else:
            random_indices = np.random.choice(self.num_grasps_,
                                              size=num_indices+1, # if i is selected
                                              replace=False).tolist()
            if i in random_indices:
                random_indices.remove(i)
            else:
                random_indices = random_indices[1:]
            random_indices = np.array(random_indices)

            X = self.X_[random_indices, :]
            S, F = self.S_[random_indices], self.F_[random_indices]

        kernels = np.array([kernel(x, xj) for xj in X])
        alphas = kernels * S
        betas = kernels * F

        alpha_i = 1 + np.sum(alphas)
        beta_i = 1 + np.sum(betas)

        v = w * np.square(X - x)

        scale = kernels * \
                ((self.mu_[i] * S / alpha_i) + \
                 ((1 - self.mu_[i]) * F / beta_i) + \
                 (S + F) / (alpha_i + beta_i))

        gradient = np.dot(scale, v)
        return gradient

    def hessian(self, w):
        raise NotImplementedError

class CrossEntropyLoss(Objective):
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p):
        self.check_valid_input(est_p)
        return -1.0 / self.N_ * np.sum((self.true_p_ * np.log(est_p) + (1.0 - self.true_p_) * np.log(1.0 - est_p)))
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class SquaredErrorLoss(Objective):
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p):
        self.check_valid_input(est_p)
        return 1.0 / self.N_ * np.sum((self.true_p_ - est_p)**2)
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class WeightedSquaredErrorLoss(Objective):
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, est_p, weights):
        self.check_valid_input(est_p)
        return np.sum(weights * (self.true_p_ - est_p)**2) * (1.0 / np.sum(weights))
    
    def check_valid_input(self, est_p):
        if not isinstance(est_p, np.ndarray):
            raise ValueError('Cross entropy must be called with ndarray')
        if est_p.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

class CCBPLogLikelihood(Objective):
    """ The log likelihood of the true params under the current distribution """
    def __init__(self, true_p):
        self.true_p_ = true_p
        self.N_ = true_p.shape[0]

    def evaluate(self, alphas, betas):
        self.check_valid_input(alphas)
        self.check_valid_input(betas)
        log_density = ss.beta.logpdf(self.true_p_, alphas, betas)
        return (1.0 / self.N_) * np.sum(log_density)
    
    def check_valid_input(self, alphas):
        if not isinstance(alphas, np.ndarray):
            raise ValueError('CCBP ML must be called with ndarray')
        if alphas.shape[0] != self.N_:
            raise ValueError('Must supply same number of datapoints as true P')

# FINITE DIFFERENCE CLASSES
class FiniteDiffableObjective(Objective):
    __metaclass__ = ABCMeta

    @abstractmethod
    def num_outputs(self):
        """ Returns the number of outputs """
        pass

class FiniteDiffableVariable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def num_dimensions(self):
        """ Returns the dimensionality of the variable """
        pass

    @abstractmethod
    def add(self, delta, i):
        """ Adds delta to the specified dimension """
        pass

class RidgeRegressionObjective(FiniteDiffableObjective):
    def __init__(self, func, Z, Y, alpha=0, params=None):
        self.f_ = func #eval function
        self.Z_ = Z #inputs
        self.Y_ = Y #targets
        self.alpha_ = alpha
        self.params_ = params

    def evaluate(self, x):
        """ L2 loss on the supplied objective """
        self.check_valid_input(x)

        loss = 0#self.alpha_ * np.linalg.norm(x)
        for z_i, y_i in zip(self.Z_, self.Y_):
            f = self.f_(z_i, x, self.params_)
            loss += (f - y_i)**2
        return loss

    def num_outputs(self):
        return 1

    def check_valid_input(self, x):
        pass

class FiniteDiffObjective(DifferentiableObjective):
    def __init__(self, objective, delta=1e-2):
        if not isinstance(objective, FiniteDiffableObjective):
            raise ValueError('Must provide finite differentiable objective')
        self.f_ = objective
        self.delta_ = delta

    def check_valid_input(self):
        if not isinstance(x, FiniteDiffableVariable):
            raise ValueError('Must provide optimizable variable')

    def gradient(self, x):
        self.check_valid_input(x)

        df_dx = np.zeros([self.f_.num_outputs(), x.num_dimensions()])
        for i in theta.num_variables:
            x_i_p = x.add( delta, i)
            x_i_m = x.add(-delta, i)
            df_dx[:,i] = (self.f_(x_i_p) - self.f_(x_i_m)) / (2 * self.delta_)

        return df_dx

    def hessian(self, x):
        raise NotImplementedError
