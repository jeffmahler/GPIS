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
from nearest_features import NearestFeatures

import sys
sys.path.insert(0, 'src/grasp_selection/')
import experiment_config as ec

def create_distance_matrix(nearest_features, feature_vectors, sorted_keys, feature_db):
	# sorted_keys = sorted_keys[:20]
	distance_function = distance.euclidean
	feature_vectors = nearest_features._project_feature_vectors(feature_vectors)

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
				try:
					dist = distance_function(feature_vectors[r_key], feature_vectors[c_key])
				except KeyError as e:
					print 'ERROR: %s' % str(e)
					dist = 1000
					
				distance_mat[r_index, c_index] = dist
				distance_mat[c_index, r_index] = dist
		print r_index

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

def compute_accuracy(nearest_features, feature_vectors, feature_db, K=1):
	UNKNOWN_TAG = 'No Results'

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

	for index, key in enumerate(feature_vectors.keys()):
		#NOTE: This is assuming the file structure is: data/<dataset_name>/<category>/... 
		query_category = mesh_db.object_category_for_key(key)
		neighbor_keys, distances = nearest_features.k_nearest_keys(feature_vectors[key], k=K+1)

		# check if top K items contains the query category
		# import IPython; IPython.embed()
		pred_category = UNKNOWN_TAG
		if len(neighbor_keys) > 0:
			pred_category = mesh_db.object_category_for_key(neighbor_keys[1])

			for i in range(1, min(K, len(neighbor_keys))):
				potential_category = mesh_db.object_category_for_key(neighbor_keys[i])

				if potential_category == query_category:
					pred_category = potential_category
					break

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
	num_preds = len(feature_vectors)
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
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('config')
	args = parser.parse_args()
	config = ec.ExperimentConfig(args.config)

	feature_db = FeatureDatabase(config)

	nearest_features = NearestFeatures(feature_db, pca_components=1000)


	compute_accuracy(nearest_features, feature_db.feature_vectors(), feature_db)

	# sorted_keys = sorted(feature_db.mesh_database().sorted_keys())

	# distance_mat = create_distance_matrix(nearest_features, feature_db.feature_vectors(), sorted_keys, feature_db)

	# distances_file = open('/Users/MelRod/Desktop/distances.txt', 'w')

	# rows, cols = distance_mat.shape
	# for r in range(0, rows):
	# 	for c in range(0, cols):
	# 		dist = distance_mat[r, c]
	# 		distances_file.write('%f\n' % (dist))
	# distances_file.close()

	



	# length = len(sorted_keys)
	# distance_mat = np.zeros((length, length))

	# print 'CREATING MATRIX'
	# lines = distances_file.readlines()
	# index = 0
	# for r in range(0, length):
	# 	for c in range(0, length):
	# 		distance_mat[r,c] = float(lines[index])
	# 		index += 1
	# 	if r%1000 == 0:
	# 		print r

	# print 'CREATING ACCURACY'
	# num_correct = 0
	# rows, cols = distance_mat.shape
	# for r in range(0, rows):
	# 	min_dist = 100000
	# 	min_index = -1
	# 	for c in range(0, cols):
	# 		if r == c:
	# 			continue
	# 		dist = distance_mat[r, c]
	# 		if dist < min_dist:
	# 			min_dist = dist
	# 			min_index = c
	# 	r_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[r])
	# 	nearest_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[min_index])
	# 	print sorted_keys[r]+'--'+r_cat+', '+sorted_keys[min_index]+'--'+nearest_cat+': '+str(min_dist)
	# 	if r_cat == nearest_cat:
	# 		num_correct += 1
	# 	# print r

	# print 'CORRECT: '+str(num_correct)
	# print 'TOTAL: '+str(len(sorted_keys))
	# print 'ACCURACY: '+str(num_correct/len(sorted_keys))
	# import IPython; IPython.embed()
