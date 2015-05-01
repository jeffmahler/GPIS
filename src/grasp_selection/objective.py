from abc import ABCMeta, abstractmethod

class Objective:
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x):
        """
        Evaluate the function at x
        """
        pass

class DifferentiableObjective(Objective):
    __metaclass__ = ABCMeta    

    @abstractmethod
    def gradient(self, x):
        """
        Evaluate the gradient at x
        """
        pass

    @abstractmethod
    def hessian(self, x):
        """
        Evaluate the hessian at x
        """
        pass

class NonDeterministicObjective(Objective):
    
    def __init__(self, det_objective):
        """
        Wraps around a deterministic objective
        """
        self.det_objective_ = det_objective

    def evaluate(self, x):
        """
        Samlpe the input space, the evaluate
        """
        if not hasattr(candidates[0], "sample"):
            raise ValueError('Input must implement a sample function')            

        x_sampled = x.sample()
        return self.det_objective_.evaluate(x)
