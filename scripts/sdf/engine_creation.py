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

import IPython

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

UNKNOWN_TAG = 'No Results'

def cat50_file_category(filename):
    """
    Returns the category associated with the full path of file |filename|
    """
    full_filename = path.abspath(filename)
    dirs, file_root = path.split(full_filename)
    head, category = path.split(dirs)
    return category

def remove_double_clean(root_dir):
    invalid_ending = "_clean_clean.sdf"
    invalid_chars = len(invalid_ending)
    num_invalid = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if len(f) >= invalid_chars and f[-invalid_chars:] == invalid_ending:
                os.remove(os.path.join(root, f))
                num_invalid += 1



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

    count = 0
    for file_ in sdf_files:
        #print file_
        if count % 100 == 0:
            print 'Converted %d files' %(count)
        converted = SDF(file_)
        converted.add_to_nearpy_engine(engine)
        count += 1
    return engine

def train_and_test_lsh(num_train, num_test, root_dir, K = 1):
    """
    Function that generates a list of sdf files given a root_dir, and loads a random num_train of them into a nearpy engine. It then queries the LSH engine for a 
    random num_test other sdf files. num_train+num_test must be less than the total number of sdf_files
    
    Parameters
        num_train: number of files to load into the engine
        num_test: number of files to query after
        sdf_files: list of sdf files to draw from
        K: number of neighbors to check

    Returns
        accuracy: float representing the accuracy of querying the nearpy engine with the test results
        engine: the trained and "tested" nearpy engine 
        test_results: dictionary of the results from the "testing" for each of the sdf_files 
    Sample Usage
        >>> train_and_test_lsh(100,5,"datasets/Cat50_ModelDatabase")
    """
    test_results = {}
    confusion = {}

    sdf_files = find_sdf(root_dir)
    print 'Found %d SDF files' %(len(sdf_files))
    assert num_train+num_test <= len(sdf_files)

    #Randomly permutes the indices of the sdf_files list. 
    np.random.seed(100)
    permuted_indices = np.random.permutation(len(sdf_files))
    get_training = itemgetter(*permuted_indices[:num_train])
    get_testing = itemgetter(*permuted_indices[num_train:num_train+num_test])
    engine = load_engine(get_training(sdf_files))
    
    if num_test > 1:
        test_files = get_testing(sdf_files)
    else:
        test_files = [get_testing(sdf_files)]

    # setup confusion matrix
    confusion[UNKNOWN_TAG] = {}
    for file_ in sdf_files:
        category = cat50_file_category(file_)
        confusion[category] = {}
    for query_cat in confusion.keys():
        for pred_cat in confusion.keys():
            confusion[query_cat][pred_cat] = 0
    
    for file_ in list(test_files):
        #NOTE: This is assuming the file structure is: data/<dataset_name>/<category>/... also line 104
        query_category = cat50_file_category(file_)
        print "Querying: %s with category %s "%(file_, query_category)
        converted = SDF(file_)
        closest_names, closest_vals = converted.query_nearpy_engine(engine)

        # check if top K items contains the query category
        pred_category = UNKNOWN_TAG
        if len(closest_names) > 0:
            closest_category = closest_names[0]
            pred_category = cat50_file_category(closest_category)

            for i in range(1, min(K, len(closest_names))):
                closest_category = closest_names[i]
                potential_category = cat50_file_category(closest_category)

                if potential_category == query_category:
                    pred_category = potential_category

        print "Result Category: %s"%(pred_category)

        confusion[query_category][pred_category] += 1
        test_results[file_]= [(closest_names, closest_vals)]
    
    # convert the dictionary to a numpy array
    row_names = confusion.keys()
    confusion_mat = np.zeros([len(row_names), len(row_names)])
    i = 0
    for query_cat in confusion.keys():
        j = 0
        for pred_cat in confusion.keys():
            confusion_mat[i,j] = confusion[query_cat][pred_cat]
            j += 1
        i += 1

    # get true positives, etc for each category
    num_preds = len(test_files)
    tp = np.diag(confusion_mat)
    fp = np.sum(confusion_mat, axis=0) - np.diag(confusion_mat)
    fn = np.sum(confusion_mat, axis=1) - np.diag(confusion_mat)
    tn = num_preds * np.ones(tp.shape) - tp - fp - fn

    # compute useful statistics
    recall = tp / (tp + fn)
    tnr = tn / (fp + tn)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    fpr = fp / (fp + tn)
    accuracy = np.sum(tp) / num_preds # correct predictions over entire dataset

    # remove nans
    recall[np.isnan(recall)] = 0
    tnr[np.isnan(tnr)] = 0
    precision[np.isnan(precision)] = 0
    npv[np.isnan(npv)] = 0
    fpr[np.isnan(fpr)] = 0

    return accuracy, engine, test_results   

    
