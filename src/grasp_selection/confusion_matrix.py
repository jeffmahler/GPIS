"""
For clean evaluation of classifier performance
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import IPython
import logging
import numpy as np
import os
import sys
import time

class ConfusionMatrix(object):
    """ Encapsulates retrieval accuracy basic structure """
    __metaclass__ = ABCMeta

    UNKNOWN_TAG = 'unknown'
    POSITIVE_TAG = 1
    NEGATIVE_TAG = 0
    
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

    def matrix(self):
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
        return confusion_mat

    def compute_performance(self):
        """ Computes all performance measures and returns in a dict, as well as stores locally """
        categories = self.matrix_.keys()
        confusion_mat = self.matrix()

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

class BinaryConfusionMatrix(ConfusionMatrix):

    def __init__(self, truth=None, predictions=None):
        if (truth is not None and predictions is None) or \
                (truth is None and predictions is not None):
            raise ValueError('Truth and predictions must both be specified')

        categories = [ConfusionMatrix.POSITIVE_TAG, ConfusionMatrix.NEGATIVE_TAG]
        ConfusionMatrix.__init__(self, categories)
        self.batch_update(truth, predictions)

    def update(self, true_category, pred_category):
        """ Considers correct for only the query category """
        # check valid category
        if pred_category != ConfusionMatrix.POSITIVE_TAG and pred_category != ConfusionMatrix.NEGATIVE_TAG:
            logging.warning('Illegal category. Skipping.')
            return

        # update matrix
        self.matrix_[pred_category][query_category] += 1

    def batch_update(self, true_categories, pred_categories):
        """ Considers correct for only the query category """
        if np.sum((pred_categories != ConfusionMatrix.POSITIVE_TAG) & (pred_categories != ConfusionMatrix.NEGATIVE_TAG)) > 0:
            logging.warning('Predictions contains illegal categories. Skipping')
            return
        if np.sum((true_categories != ConfusionMatrix.POSITIVE_TAG) & (true_categories != ConfusionMatrix.NEGATIVE_TAG)) > 0:
            logging.warning('Truth contains illegal categories. Skipping')
            return

        fp = np.sum((pred_categories == ConfusionMatrix.POSITIVE_TAG) & (true_categories == ConfusionMatrix.NEGATIVE_TAG))
        tp = np.sum((pred_categories == ConfusionMatrix.POSITIVE_TAG) & (true_categories == ConfusionMatrix.POSITIVE_TAG))
        fn = np.sum((pred_categories == ConfusionMatrix.NEGATIVE_TAG) & (true_categories == ConfusionMatrix.POSITIVE_TAG))
        tn = np.sum((pred_categories == ConfusionMatrix.NEGATIVE_TAG) & (true_categories == ConfusionMatrix.NEGATIVE_TAG))
        self.matrix_[ConfusionMatrix.NEGATIVE_TAG][ConfusionMatrix.POSITIVE_TAG] += fp
        self.matrix_[ConfusionMatrix.POSITIVE_TAG][ConfusionMatrix.POSITIVE_TAG] += tp
        self.matrix_[ConfusionMatrix.POSITIVE_TAG][ConfusionMatrix.NEGATIVE_TAG] += fn
        self.matrix_[ConfusionMatrix.NEGATIVE_TAG][ConfusionMatrix.NEGATIVE_TAG] += tn

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
