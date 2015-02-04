"""
Functions to grab all sdf files and put them in a nearpy engine
Author: Sahaana Suri
"""

from os import walk, path
from sdf_class import SDF
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


def find_sdf(root_dir):
    """
    Function that traverses the directory tree beginning with root_dir, and finds all of the SDF files contained within

    Parameters 
        root_dir: string that represents the directory in which you want to begin searching (relative to the current directory, or absolute)

    Returns
        sdf_files: a list of path names (relative to root_dir) to all sdf files found under the root_dir

    Sample Usage
        >>> sdf_files =  find_sdf("datasets") 
    """
    sdf_files = []
    for root,dirs,files in walk(root_dir):
        for file_ in files:
            if file_.endswith(".sdf"):
                sdf_files.append(path.join(root,file_))
    return sdf_files

def load_engine(sdf_files):
    """
    Function that converts the given sdf_files into instances of the sdf_class, then loads them into nearpy Engine.

    Parameters
        sdf_files: a list of sdf_files with their pathname from the current directory. Intended to be fed in from `find_sdf(root_dir)`

    Returns
        engine: instance of a nearpy engine with all of sdf_files loaded
    
    Sample Usage
        >>> engine = load_engine(sdf_files)
    """
    #Can be made more general by querying the dimensions of the SDF. Or could even be fed into the function if I make an additional wrapper
    dimension = 50*50*50
    #dimension here can be altered as well
    rbp = RandomBinaryProjections('rbp',10)
    engine = Engine(dimension, lshashes=[rbp])  
    for file_ in sdf_files[:10]:
        converted = SDF(file_)
        converted.add_to_nearpy_engine(engine)
    return engine

def train_and_test_lsh(num_train, num_test, root_dir):
    """
    Function that generates a list of sdf files given a root_dir, and loads a random num_train of them into a nearpy engine. It then queries the LSH engine for a 
    random num_test other sdf files. num_train+num_test must be less than the total number of sdf_files
    
    Parameters
        num_train: number of files to load into the engine
        num_test: number of files to query after
        sdf_files: list of sdf files to draw from

    Returns
        engine: the trained and "tested" nearpy engine 
        test_results: vector of the results from the "testing"
    """
    test_results = []
    sdf_files = find_sdf(root_dir)
    assert num_train+num_test <= len(sdf_files)
    #Randomly permutes the indices of the sdf_files list. 
    permuted_indices = np.random.permutation(len(sdf_files))
    get_training = itemgetter(*permuted_indices[:num_train])
    get_testing = itemgetter(*permuted_indices[num_train:num_train+num_test])

    engine = load_engine(get_training(sdf_files))
    for file_ in get_testing(sdf_files):
        converted = SDF(file_)
        closest = converted.query_nearpy_engine(engine)
        print closest
        test_results.append([converted.file_name(),closest])
    return engine, test_results
        

    
