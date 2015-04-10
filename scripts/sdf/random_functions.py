"""
Functions to import for sdf class/maybe other things. Avoiding potential clutter

Author: Sahaana Suri
"""
from os import walk, path
import numpy as np


def crosses_threshold(threshold):
    def crosses(elems):
        """ 
        Function to determine if a np array has values both above and below a threshold (moved edge). Generalized "has positive and negative" function. 
        For use with filter(-,-). 
        Params: 
            elems: np array of numbers
            threshold: any number used as a threshold
        Returns: 
            (bool): True if elems has both negative and positive numbers, False otherwise
        """
        return (elems>threshold).any() and (elems<threshold).any()
    return crosses


"""
def crosses_threshold(elems, threshold):
    return (elems>threshold).any() and (elems<threshold).any()
"""



def find_sdf(root_dir, ending):
    """
    Function that traverses the directory tree beginning with root_dir, and finds all of the SDF files contained within

    Parameters 
        root_dir: string that represents the directory in which you want to begin searching (relative to the current directory, or absolute)
        ending : string for the file ending you're searching for
    Returns
        sdf_files: a list of path names (relative to root_dir) to all sdf files found under the root_dir

    Sample Usage
        >>> sdf_files =  find_sdf("datasets") 
    """
    sdf_files = []
    for root,dirs,files in walk(root_dir):
        for file_ in files:
            if file_.endswith(ending):
                sdf_files.append(path.join(root,file_))
    return sdf_files

def histogramize(Kmeans_vector, K):
    """
    Function that takes in the vector returned by Kmean and normalizes. 

    Parameters
        Kmeans_vector: vector representing the clusters that Kmeans assigned to each entry
        K: integer representing the number of clusters used by the Kmeans algorithm

    Returns
        numpy array that is a normalized histogram form of the Kmeans vector (to be used as a feature vector)
    """
    counts = np.bincount(Kmeans_vector, minlength=K)
    return np.array([counts / float(sum(counts))])


