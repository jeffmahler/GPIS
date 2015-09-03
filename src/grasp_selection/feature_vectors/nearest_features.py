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
from scipy.spatial import distance

from feature_database import FeatureDatabase
from dataset_sorter import DatasetSorter

import sys
sys.path.append('src/grasp_selection/')
import kernels

class FeatureObject:
	def __init__(self, key, feature_vector):
		self.key = key
		self.feature_vector = feature_vector

class NearestFeatures:
	def __init__(self, feature_db, pca_components=10, feature_vectors=None, svd=None, neighbor_tree=None, neighbor_data=None):
		if feature_vectors == None:
			feature_vectors = feature_db.feature_vectors() #feature_db.train_feature_vectors()

		self.pca_components = pca_components
		if svd == None:
			self.svd = self._create_train_svd(feature_vectors, pca_components)
			data_size = len(feature_vectors.keys())
			if self.pca_components > data_size:
				self.pca_components = data_size
		else:
			self.svd = svd

		if neighbor_tree == None:
			start = time.time()
			data = self._project_feature_vectors(feature_vectors)
			self.neighbors = self._create_neighbors(data)
			end = time.time()
			print 'TIME: %0.4f' % (end - start)
		else:
			self.neighbors = kernels.KDTree(phi=lambda x: x.feature_vector)
			self.neighbors.tree_ = neighbor_tree
			self.neighbors.data_ = neighbor_data

	def within_distance(self, feature_vector, dist=0.5):
		feature_object = self._create_query_object(feature_vector)
		neighbor_feature_objects, distances = self.neighbors.within_distance(feature_object, dist=dist)
		return self._create_feature_vector_dict(neighbor_feature_objects)

	def k_nearest(self, feature_vector, k=1):
		feature_object = self._create_query_object(feature_vector)
		neighbor_feature_objects, distances = self.neighbors.nearest_neighbors(feature_object, k)
		return self._create_feature_vector_dict(neighbor_feature_objects)

	def k_nearest_keys(self, feature_vector, k=1):
		feature_object = self._create_query_object(feature_vector)
		neighbor_feature_objects, distances = self.neighbors.nearest_neighbors(feature_object, k)
		return map(lambda x: x.key, neighbor_feature_objects), distances

	def _create_query_object(self, feature_vector):
		if feature_vector.shape[0] != self.pca_components:
			feature_vector = self.project_feature_vector(feature_vector)
		feature_object = FeatureObject('query_object', feature_vector)
		return feature_object

	def _create_feature_vector_dict(self, feature_objects):
		keys = map(lambda x: x.key, feature_objects)
		values = map(lambda x: x.feature_vector, feature_objects)
		return dict(zip(keys, values))

	def project_feature_vector(self, feature_vector):
		X = sp_mat(feature_vector)
		return self.svd.transform(X)[0]

	def _create_train_svd(self, train_feature_vectors, pca_components):
		X = map(sp_mat, train_feature_vectors.values())
		print 'Creating SVD...'
		start = time.time()
		svd = TruncatedSVD(n_components=pca_components)
		svd.fit(vstack(X))
		print 'DONE'
		end = time.time()
		print end - start
		print 'Explained variance ratio %f' % np.sum(svd.explained_variance_ratio_)
		return svd

	def _project_feature_vectors(self, feature_vectors):
		X = map(sp_mat, feature_vectors.values())
		projected_feature_vectors = {}
		for key, vector in zip(feature_vectors.keys(), self.svd.transform(vstack(X))):
			projected_feature_vectors[key] = vector
		return projected_feature_vectors

	def _create_neighbors(self, feature_vectors):
		feature_objects = map(FeatureObject, feature_vectors.keys(), feature_vectors.values())
		neighbors = kernels.KDTree(phi=lambda x: x.feature_vector)
		neighbors.train(feature_objects)
		return neighbors

	def create_distance_matrix(self, feature_vectors, sorted_keys, feature_db):
		# sorted_keys = sorted_keys[:20]
		distance_function = distance.euclidean
		feature_vectors = self._project_feature_vectors(feature_vectors)

		length = len(sorted_keys)
		distance_mat = np.zeros((length, length))
		for r_index, r_key in enumerate(sorted_keys):
			distance_vect = []
			for c_index, c_key in enumerate(sorted_keys):
				if r_index > c_index:
					continue;
				if r_index == c_index:
					dist = 0
				else:
					if r_key in feature_vectors and c_key in feature_vectors:
						dist = distance_function(feature_vectors[r_key], feature_vectors[c_key])
					else:
						dist = 100
					distance_mat[r_index, c_index] = dist
					distance_mat[c_index, r_index] = dist
					# r_cat = feature_db.mesh_database().object_category_for_key(r_key)
					# c_cat = feature_db.mesh_database().object_category_for_key(c_key)
					# print r_key+'--'+r_cat+', '+c_key+'--'+c_cat+': '+str(dist)
			print r_index
			# print ''

		num_correct = 0
		rows, cols = distance_mat.shape
		for r in range(0, rows):
			min_dist = 100000
			min_index = -1
			for c in range(0, cols):
				if r == c:
					continue
				dist = distance_mat[r, c]
				if dist < min_dist:
					min_dist = dist
					min_index = c
			r_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[r])
			c_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[c])
			if r_cat == c_cat:
				num_correct += 1

		print 'CORRECT: '+str(num_correct)
		print 'TOTAL: '+str(len(sorted_keys))
		print 'ACCURACY: '+str(num_correct/len(sorted_keys))

		return distance_mat

	def compute_accuracy(self, feature_db, K=1):
		test_feature_vectors = self._project_feature_vectors(feature_db.feature_vectors())
		mesh_db = feature_db.mesh_database()
		categories = Set(mesh_db.object_dict().values())

		confusion = {}
		# setup confusion matrix
		confusion[UNKNOWN_TAG] = {}
		for category in categories:
			confusion[category] = {}
		for query_cat in confusion.keys():
			for pred_cat in confusion.keys():
				confusion[query_cat][pred_cat] = 0

		for index, key in enumerate(test_feature_vectors):
			#NOTE: This is assuming the file structure is: data/<dataset_name>/<category>/... 
			query_category = mesh_db.object_category_for_key(key)
			print "Querying: %s with category %s "%(key, query_category)
			query_feature_object = FeatureObject(key, test_feature_vectors[key])
			neighbors, distances = self.neighbors.nearest_neighbors(query_feature_object, K+1)
			neighbors = neighbors.flatten()

			# check if top K items contains the query category
			# import IPython; IPython.embed()
			pred_category = UNKNOWN_TAG
			if len(neighbors) > 0:
				pred_category = mesh_db.object_category_for_key(neighbors[1].key)

				for i in range(1, min(K, len(neighbors))):
					potential_category = mesh_db.object_category_for_key(neighbors[i].key)

					if potential_category == query_category:
						pred_category = potential_category
						break

			print "Result Category: %s, %s, %d"%(pred_category, neighbors[1].key, len(neighbors))

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
		num_preds = len(test_feature_vectors)
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

def test_nearest_features(nearest_features, feature_db):
	print 'TESTING NF...'
	distance_matrix = nearest_features.create_distance_matrix(feature_db.feature_vectors(), feature_db.mesh_database().sorted_keys(), feature_db)

	distances_file = open('/home/jmahler/mel/GPIS_data/data/distances.txt', 'w')
	rows, cols = distance_matrix.shape
	for r in range(0, rows):
		for c in range(0, cols):
			distances_file.write(str(distance_matrix[r, c])+'\n')
	distances_file.close()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('config')
	args = parser.parse_args()
	config = ec.ExperimentConfig(args.config)

	feature_db = feature_database.FeatureDatabase(config)

	import IPython; IPython.embed()
	# nearest_features = NearestFeatures(feature_db, pca_components=10)
	# feature_db.save_nearest_features(nearest_features)

	# test_nearest_features(nearest_features, feature_db)
	# nearest_features.compute_accuracy(feature_db)

