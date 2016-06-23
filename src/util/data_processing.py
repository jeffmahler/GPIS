'''
Utility functions to help process data
Author: Jacky
'''
import numpy as np
import logging

class DataProcessing:

    @staticmethod
    def is_1d(data):
        return data.reshape(-1, 1).shape[0] == data.shape[0]

    @staticmethod
    def nrmlz_0_to_1(data):
        '''
        Currently assumes data has to be a 1-d array
        TODO: implement arbitrary axis
        '''
        data = np.array(data)
        if not DataProcessing.is_1d(data):
            logging.error("Given data array is not one dimensional. Normalize 0 to 1 only supports 1d arrays.")
            return
            
        min_val = np.min(data)
        max_val = np.max(data)
        
        range_val = max_val - min_val
        
        data -= min_val
        data = data /1./ range_val
        
        return data
        
    @staticmethod
    def nrmlz_std_gaussian(data):
        '''
        Currently assumes data has to be a 1-d array
        TODO: implement arbitrary axis
        '''
        data = np.array(data)
        if not DataProcessing.is_1d(data):
            logging.error("Given data array is not one dimensional. Normalize 0 to 1 only supports 1d arrays.")
            return
            
        mean = np.mean(data)
        std = np.std(data)
        
        data -= mean
        data = data /1./ std
        
        return data