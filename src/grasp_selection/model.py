from abc import ABCMeta, abstractmethod

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
        Predict the a function of the data at some point x
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

    
    

