import time

from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import vstack
from sets import Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
import numpy as np
from numpy import argmin
from sklearn.decomposition import TruncatedSVD

from feature_database import FeatureDatabase
from feature_object import FeatureObject
from feature_object import FeatureObjectDatabase
from dataset_sorter import DatasetSorter

UNKNOWN_TAG = 'No Results'

class NearestFeatures:
	def __init__(self, feature_db, pca_components=10):
		train_feature_objects = feature_db.train_feature_objects()
		self.pca_components = pca_components
		self.svd = self._create_train_svd(train_feature_objects, pca_components)
		self.neighbors = self._create_neighbors(self._project_feature_objects(self.svd, train_feature_objects))

	def within_distance(self, feature_object, dist=0.5):
		feature_vector = self.svd.transform(feature_object.feature_vector)
		return neighbors.within_distance(feature_vector, dist=dist)

	def _create_train_svd(self, train_feature_objects, pca_components):
		X = map(lambda x: x.feature_vector, train_feature_objects)
		print 'Creating SVD...'
		start = time.time()
		svd = TruncatedSVD(n_components=pca_components)
		svd.fit(vstack(X))
		print 'DONE'
		end = time.time()
		print end - start
		return svd

	def _project_feature_objects(self, svd, feature_objects):
		X = map(lambda x: x.feature_vector, feature_objects)
		feature_vectors = svd.transform(vstack(X))
		feature_objects = map(lambda obj, vect: FeatureObject(obj.name, vect), feature_objects, feature_vectors)
		return feature_objects

	def _create_neighbors(self, pca_feature_objects):
		neighbors = kernels.KDTree(phi=lambda x: x.feature_vector)
		neighbors.train(feature_objects)
		return neighbors

	def compute_accuracy(self, test_feature_objects, object_database, K=1):
		confusion = {}
		# setup confusion matrix
		confusion[UNKNOWN_TAG] = {}
		for category in categories:
			confusion[category] = {}
		for query_cat in confusion.keys():
			for pred_cat in confusion.keys():
				confusion[query_cat][pred_cat] = 0

		for index, feature_object in enumerate(self.feature_objects):
			#NOTE: This is assuming the file structure is: data/<dataset_name>/<category>/... 
			query_category = name_to_category[feature_object.name]
			print "Querying: %s with category %s "%(feature_object.name, query_category)
			neighbors, distances = neighbors.nearest_neighbors(feature_object, K)
			neighbors = neighbors.flatten()

			# check if top K items contains the query category
			pred_category = UNKNOWN_TAG
			if len(neighbors) > 0:
				pred_category = name_to_category[neighbors[0].name]

				for i in range(0, min(K, len(neighbors))):
					potential_category = name_to_category[neighbors[i].name]

					if potential_category == query_category:
						pred_category = potential_category
						break

			print "Result Category: %s, %d"%(pred_category, len(neighbors))

			confusion[query_category][pred_category] += 1

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
		num_preds = len(self.feature_objects)
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

		print 'FINAL ACCURACY: '+str(accuracy)
		stats = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'tnr':tnr, 'npv':npv, 'fpr':fpr}

		return confusion, stats

if __name__ == '__main__':
	feature_db = FeatureDatabase()
	nearest_features = NearestFeatures(feature_db)
	feature_db.save_nearest_features(feature_object_db)


