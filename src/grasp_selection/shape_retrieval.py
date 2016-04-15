"""
Script for running shape retrieval experiments
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import IPython
import logging
import numpy as np
import os
import sys
import time
sys.path.append('src/grasp_selection/feature_vectors/')

import sklearn.decomposition as skd
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import vstack

import database as db
import experiment_config as ec
import kernels
import mesh_database as md
import mvcnn_feature_extractor as mvcnn

SUPPORTED_DATASETS = ['ModelNet40', 'SHREC14LSGTB'] # list of datasets that can be run

class FakeGraspable(object):
    """ Struct to spoof graspables for faster testing """ 
    def __init__(self, key):
        self.key = key

class ConfusionMatrix(object):
    """ Encapsulates retrieval accuracy basic structure """
    __metaclass__ = ABCMeta

    UNKNOWN_TAG = 'unknown'
    
    def __init__(self, categories):
        self._setup_matrix(categories)
        
    def _setup_matrix(self, categories):
        """ Make '2D' dictionary with counts of number of times a query (row) was predicted as a category (col) """ 
        self.matrix_ = {}
        self.matrix_[ConfusionMatrix.UNKNOWN_TAG] = {}
        for category in categories:
            self.matrix_[category] = {}
        for query_cat in self.matrix_.keys():
            for pred_cat in self.matrix_.keys():
                self.matrix_[query_cat][pred_cat] = 0
        
    @abstractmethod
    def update(self, query_category, retrieved_categories):
        """ Updates counts based on whether the retrieved category contains the desired category """
        pass

    def compute_performance(self):
        """ Computes all performance measures and returns in a dict, as well as stores locally """
        # compile confusion dict into a matrix
        categories = self.matrix_.keys()
        confusion_mat = np.zeros([len(categories), len(categories)])
        i = 0
        for query_cat in categories:
            j = 0
            for pred_cat in categories:
                confusion_mat[i,j] = self.matrix_[query_cat][pred_cat]
                j += 1
            i += 1

        # get ratio positives, negatives for each category
        num_preds = np.sum(confusion_mat)
        tp = np.diag(confusion_mat)
        fp = np.sum(confusion_mat, axis=0) - np.diag(confusion_mat)
        fn = np.sum(confusion_mat, axis=1) - np.diag(confusion_mat)
        tn = num_preds * np.ones(tp.shape) - tp - fp - fn
        all_tp = np.sum(tp)
        all_fp = np.sum(fp)
        all_fn = np.sum(fn)
        all_tn = np.sum(tn)

        # compute useful statistics
        recall = tp / (tp + fn)
        tnr = tn / (fp + tn)
        precision = tp / (tp + fp)
        npv = tn / (tn + fn)
        fpr = fp / (fp + tn)
        accuracy = tp / num_preds # correct predictions over entire dataset
        all_recall = all_tp / (all_tp + all_fn)
        all_tnr = all_tn / (all_fp + all_tn)
        all_precision = all_tp / (all_tp + all_fp)
        all_npv = all_tn / (all_tn + all_fn)
        all_fpr = all_fp / (all_fp + all_tn)
        all_accuracy = all_tp / num_preds # correct predictions over entire dataset

        # remove nans
        recall[np.isnan(recall)] = 0
        tnr[np.isnan(tnr)] = 0
        precision[np.isnan(precision)] = 0
        npv[np.isnan(npv)] = 0
        fpr[np.isnan(fpr)] = 0
        
        # create dictionary of useful statistics
        holistic_metrics = {'accuracy':all_accuracy, 'precision':all_precision, 'recall':all_recall, 'tnr':all_tnr, 'npv':all_npv, 'fpr':all_fpr,
                            'tp': all_tp, 'fp': all_fp, 'fn':all_fn, 'tn':all_tn, 'categories':categories}
        per_class_metrics = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'tnr':tnr, 'npv':npv, 'fpr':fpr,
                             'tp': tp, 'fp': fp, 'fn':fn, 'tn':tn, 'categories':categories}
        return holistic_metrics, per_class_metrics

class TopKConfusionMatrix(ConfusionMatrix):
    def __init__(self, categories, K=1):
        self.K_ = K
        ConfusionMatrix.__init__(self, categories)

    def update(self, query_category, retrieved_categories):
        """ Considers correct if any retrieved_categories contains the query """
        if len(retrieved_categories) > self.K_:
            logging.error('Too many retrieved categories. Ignoring update')
            return

        # set the initial prediction
        pred_category = ConfusionMatrix.UNKNOWN_TAG
        if len(retrieved_categories) > 0:
            pred_category = retrieved_categories[0]

        # search for the query cateogry
        for ret_cat in retrieved_categories:
            if ret_cat == query_category:
                pred_category = query_category
                break

        # update matrix
        self.matrix_[query_category][pred_category] += 1

class MajorityKConfusionMatrix(ConfusionMatrix):
    def __init__(self, categories, K=1):
        self.K_ = K
        ConfusionMatrix.__init__(self, categories)

    def update(self, query_category, retrieved_categories):
        """ Considers correct if the majority vote in retrieved_categories is the query """
        if len(retrieved_categories) > self.K_:
            logging.error('Too many retrieved categories. Ignoring update')
            return

        # set the initial prediction
        cat_counts = {}
        for key in retrieved_categories:
            cat_counts[key] = 0
        for ret_cat in retrieved_categories:
            cat_counts[ret_cat] += 1

        # find the max
        max_count = 0
        pred_cat = ConfusionMatrix.UNKNOWN_TAG
        for cat, count in cat_counts.iteritems():
            if count > max_count:
                max_count = count
                pred_cat = cat
        
        # update matrix
        self.matrix_[query_category][pred_cat] += 1

class TopNConfusionMatrix(ConfusionMatrix):
    def __init__(self, categories):
        ConfusionMatrix.__init__(self, categories)

    def update(self, query_category, retrieved_categories):
        """ Considers correct if any retrieved_categories contains the query """
        pred_category = ConfusionMatrix.UNKNOWN_TAG
        for ret_cat in retrieved_categories:
            if ret_cat == query_category:
                pred_category = query_category
                self.matrix_[query_category] += 1

def train_svd(features, num_components=10):
    feature_descriptors = [f.descriptor for f in features]
    X = map(sp_mat, feature_descriptors)
    start = time.time()
    logging.info('Computing truncated SVD')
    svd = skd.TruncatedSVD(n_components=num_components)
    svd.fit(vstack(X))
    end = time.time()
    logging.info('Truncated SVD took %f sec' %(end - start))
    logging.info('Explained variance ratio %f' % np.sum(svd.explained_variance_ratio_))
    return svd

def project_feature_vectors(feature_vectors, projector):
    X = map(sp_mat, feature_vectors.values())
    projected_feature_vectors = []
    for vector in projector.transform(vstack(X)):
        projected_feature_vectors.append(vector)
    return projected_feature_vectors

if __name__ == '__main__':
    # read and setup config
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(100) # make deterministic for now
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)
    K = config['retrieval']['num_neighbors']
    
    # check valid dataset option
    if config['dataset'] not in SUPPORTED_DATASETS:
        logging.error('Dataset %s not supported at this time. Exiting...' %(config['dataset']))
        exit(0)
    if config['dataset'] == 'SHREC14LSGTB':
        path_to_class_file = os.path.join(config['database_dir'], config['dataset'], 'SHREC14LSGTB.cla')
        object_database = md.SHRECObjectDatabase(path_to_class_file)
    else:
        path_to_class_file = os.path.join(config['database_dir'], config['dataset'], 'index.db')
        object_database = md.ModelNet40ObjectDatabase(path_to_class_file)        

    # load data
    dataset = db.FilesystemChunk(config)
    graspables = [FakeGraspable(config['dataset'] + '_' + k) for k in dataset.data_keys] # assumed in the big keys file

    # break into training / test
    num_datapoints = len(graspables)
    training_pct = config['retrieval']['train_pct']
    num_training = int(num_datapoints * training_pct)
    training_indices = np.random.choice(num_datapoints, size=num_training, replace=False)
    test_indices = np.setdiff1d(np.arange(num_datapoints), training_indices)
    training_graspables = [graspables[i] for i in training_indices.tolist()]
    test_graspables = [graspables[i] for i in test_indices.tolist()]

    # create MVCNN descriptors for training
    if config['retrieval']['train_extraction_method'] == 'subset':
        training_extractor = mvcnn.MVCNNSubsetBatchFeatureExtractor(config['retrieval']['num_images'], config)
    elif config['retrieval']['train_extraction_method'] == 'image':
        training_extractor = mvcnn.MVCNNImageBatchFeatureExtractor(config)
    else:
        training_extractor = mvcnn.MVCNNBatchFeatureExtractor(config)        
    training_features = training_extractor.extract(training_graspables)
    svd = train_svd(training_features, num_components=config['retrieval']['num_components'])

    # add to a nearest neighbors struct
    training_keys = [f.key for f in training_features]
    training_descriptors = [f.descriptor for f in training_features]
    training_sparse_descriptors = map(sp_mat, training_descriptors)
    training_key_feature_pairs = zip(training_keys, training_sparse_descriptors)
    phi = lambda x : svd.transform(x[1])
    nn_struct = kernels.KDTree(phi=phi)
    nn_struct.train(training_key_feature_pairs)
    
    # create test features
    all_one_nn_holistic_metrics = []
    all_holistic_metrics = []

    for test_extraction_method, test_num_images in zip(config['retrieval']['test_extraction_methods'], config['retrieval']['test_num_images']):
        if test_extraction_method == 'subset':
            test_extractor = mvcnn.MVCNNSubsetBatchFeatureExtractor(test_num_images, config)
        elif test_extraction_method == 'image':
            test_extractor = mvcnn.MVCNNImageBatchFeatureExtractor(config)
        else:
            test_extractor = mvcnn.MVCNNBatchFeatureExtractor(config)        
        test_features = test_extractor.extract(test_graspables) 
        test_keys = [f.key for f in test_features]
        test_descriptors = [f.descriptor for f in test_features]
        test_sparse_descriptors = map(sp_mat, test_descriptors)
        test_key_feature_pairs = zip(test_keys, test_sparse_descriptors)

        # generate predictions and create matrices
        all_cats = set(object_database.object_dict_.values())
        one_nn_confusion = TopKConfusionMatrix(all_cats, K=1)
        confusion = TopKConfusionMatrix(all_cats, K=K)
        for test_key_feature_pair in test_key_feature_pairs:
            query_cat = object_database.object_category_for_key(test_key_feature_pair[0])
            neighbors, distances = nn_struct.nearest_neighbors(test_key_feature_pair, K)
            retrieved_cats = [object_database.object_category_for_key(n[0]) for n in neighbors]
            confusion.update(query_cat, retrieved_cats)
            one_nn_confusion.update(query_cat, retrieved_cats[0:1])
            
        # compute and print metrics
        holistic_metrics, per_class_metrics = confusion.compute_performance()
        logging.info('')
        logging.info('OVERALL TOP-%d PERFORMANCE' %(K))
        for metric, value in holistic_metrics.iteritems():
            if metric != 'categories':
                logging.info('%s: %f' %(metric, value))

        one_nn_holistic_metrics, one_nn_per_class_metrics = one_nn_confusion.compute_performance()
        logging.info('')
        logging.info('OVERALL 1-NN PERFORMANCE')
        for metric, value in one_nn_holistic_metrics.iteritems():
            if metric != 'categories':
                logging.info('%s: %f' %(metric, value))

        all_holistic_metrics.append(holistic_metrics)
        all_one_nn_holistic_metrics.append(one_nn_holistic_metrics)

    for holistic_metrics, one_nn_holistic_metrics, num_images in zip(all_holistic_metrics, all_one_nn_holistic_metrics, config['retrieval']['test_num_images']):
        logging.info('')
        logging.info('USING %d IMAGES' %(num_images))
        logging.info('TOP-%d PERFORMANCE' %(K))
        for metric, value in holistic_metrics.iteritems():
            if metric != 'categories':
                logging.info('%s: %f' %(metric, value))

        logging.info('')
        logging.info('1-NN PERFORMANCE')
        for metric, value in one_nn_holistic_metrics.iteritems():
            if metric != 'categories':
                logging.info('%s: %f' %(metric, value))

    IPython.embed()
    
