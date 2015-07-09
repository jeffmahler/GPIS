import os
import pickle
import time
import yaml

from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import vstack
# 
from sets import Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
import numpy as np
from numpy import argmin
from sklearn.decomposition import TruncatedSVD

config = yaml.load(open('config/config.yaml', 'r'))
gpis_root = config['gpis_root']
import sys
sys.path.insert(0, gpis_root+'src/grasp_selection')
import kernels

data_set = 'Cat 50'
image_type = 'color'
layer = 'fc7'

path_to_data_dir = 'data/'
path_to_index_file = path_to_data_dir+'Cat50_ModelDatabase_index.db'
path_to_save_dir = 'feature_trials/'

path_to_trial = 'caffe_trials/8/'
training_objects_file = path_to_trial+'training_objects.txt'
test_objects_file = path_to_trial+'test_objects.txt'
pickle_file_name = 'feature_objects_'+image_type+'_'+layer+'.p'

K = 1
PCA_components = 2
UNKNOWN_TAG = 'No Results'

class FeatureObject:
	def __init__(self, name, feature_vector):
		self.name = name
		self.feature_vector = feature_vector

def object_names_from_file(path_to_objects_file):
	object_names = []
	with open(path_to_objects_file) as file:
		for line in file:
			object_names.append(line.split()[0])
	return object_names

def create_name_to_category(path_to_file):
	name_to_category = {}
	categories = Set()
	with open(path_to_file) as file:
		for line in file:
			split = line.split()
			file_name = split[0]
			category = split[1]
			name_to_category[file_name] = category
			categories.add(category)
	return name_to_category, categories

def get_feature_objects_from_file(file):
	feature_objects = pickle.load(open(file, "rb"))
	return feature_objects

def create_and_train_nearpy(feature_objects):
	start = time.clock()
	# Nearest Neighbors:
	nearpy = kernels.NearPy(phi=lambda x: x.feature_vector)
	nearpy.train(feature_objects)
	end = time.clock()
	print end - start
	return nearpy

def create_and_train_kdtree(feature_objects):
	start = time.clock()
	# Nearest Neighbors:
	kdtree = kernels.KDTree(phi=lambda x: x.feature_vector)
	kdtree.train(feature_objects)
	end = time.clock()
	print end - start
	return kdtree

def compute_accuracy(feature_objects, name_to_category, categories, kdtree):
	confusion = {}
	# setup confusion matrix
	confusion[UNKNOWN_TAG] = {}
	for category in categories:
		confusion[category] = {}
	for query_cat in confusion.keys():
		for pred_cat in confusion.keys():
			confusion[query_cat][pred_cat] = 0

	for index, feature_object in enumerate(feature_objects):
		#NOTE: This is assuming the file structure is: data/<dataset_name>/<category>/... 
		query_category = name_to_category[feature_object.name]
		print "Querying: %s with category %s "%(feature_object.name, query_category)
		neighbors, distances = kdtree.nearest_neighbors(feature_object, K)
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
	num_preds = len(feature_objects)
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
	write_stats_to_file(stats, path_to_save_dir, confusion_mat)

def write_stats_to_file(stats, save_dir, confusion_mat):
	name_index = 0
	dir = path_to_save_dir+str(name_index)+'/'
	while os.path.isdir(dir):
		name_index += 1
		dir = path_to_save_dir+str(name_index)+'/'
	os.makedirs(dir)
	with open(dir+'stats.txt', 'w') as file:
		for stat_names in stats.keys():
			file.write(stat_names+': '+str(stats[stat_names])+'\n')
	with open(dir+'data.txt', 'w') as file:
		file.write('DATA SET: '+data_set+'\n')
		file.write('RENDER TYPE: '+image_type+'\n')
		file.write('LAYER: '+layer+'\n')
		file.write('K: '+str(K)+'\n')
	pickle.dump(confusion_mat, open(dir+'confusion.p', "wb"))

def create_train_svd(feature_objects):
	X = map(lambda x: x.feature_vector, feature_objects)

	print 'Creating SVD...'
	start = time.time()
	svd = TruncatedSVD(n_components=PCA_components)
	svd.fit(vstack(X))
	print 'DONE'
	end = time.time()
	print end - start
	return svd

def transform_feature_objects(feature_objects, svd):
	X = map(lambda x: x.feature_vector, feature_objects)
	feature_vectors = svd.transform(vstack(X))
	feature_objects = map(lambda obj, vect: FeatureObject(obj.name, vect), feature_objects, feature_vectors)
	return feature_objects

def scatter_feature_objects(plt, feature_objects, color):
	x = map(lambda obj: obj.feature_vector[0], feature_objects)
	y = map(lambda obj: obj.feature_vector[1], feature_objects)
	return plt.scatter(x, y, s=50, c=color)

def plot_feature_objects_data(feature_objects, name_to_category, categories):
	objs = []
	labels = ['Others']
	plt.figure()
	colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(categories)))
	objs.append(scatter_feature_objects(plt, training_feature_objects+test_feature_objects, '#eeeeff'))

	i = 0
	for category in categories:
		filter_for_cat = lambda obj: name_to_category[obj.name] == category
		sub_fobjs = filter(filter_for_cat, training_feature_objects)
		objs.append(scatter_feature_objects(plt, sub_fobjs, colors[i]))
		labels.append(category)
		i += 1

	plt.legend(objs, labels)
	plt.show()

def plot_training_vs_test_data(training_feature_objects, test_feature_objects, name_to_category, categories):
	objs = []
	labels = ['Others']
	plt.figure()
	colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(categories)*2))
	objs.append(scatter_feature_objects(plt, training_feature_objects+test_feature_objects, '#eeeeff'))

	i = 0
	for category in categories:
		filter_for_cat = lambda obj: name_to_category[obj.name] == category
		sub_training_fobjs = filter(filter_for_cat, training_feature_objects)
		sub_test_fobjs = filter(filter_for_cat, test_feature_objects)
		objs.append(scatter_feature_objects(plt, sub_training_fobjs, colors[i]))
		objs.append(scatter_feature_objects(plt, sub_test_fobjs, colors[i+1]))
		labels.extend([category+'_training', category+'_test'])
		i += 2

	plt.legend(objs, labels)
	plt.show()

def plot_data_with_indices(training_feature_objects, test_feature_objects, name_to_category, cat_list, indices, train_vs_test=False):
	categories = map(lambda index: cat_list[index], indices)
	if train_vs_test:
		plot_training_vs_test_data(training_feature_objects, test_feature_objects, name_to_category, categories)
	else:
		plot_feature_objects_data(training_feature_objects+test_feature_objects, name_to_category, categories)

feature_objects = get_feature_objects_from_file(path_to_data_dir+pickle_file_name)
training_object_names = object_names_from_file(training_objects_file)
test_object_names = object_names_from_file(test_objects_file)
training_feature_objects = filter(lambda o: o.name in training_object_names, feature_objects)
test_feature_objects = filter(lambda o: o.name in test_object_names, feature_objects)
print len(training_feature_objects)
print len(test_feature_objects)

svd = create_train_svd(training_feature_objects)
# import IPython; IPython.embed()
training_feature_objects = transform_feature_objects(training_feature_objects, svd)
test_feature_objects = transform_feature_objects(test_feature_objects, svd)

name_to_category, categories = create_name_to_category(path_to_index_file)
kdtree = create_and_train_kdtree(training_feature_objects)
compute_accuracy(test_feature_objects, name_to_category, categories, kdtree)
plot_data_with_indices(training_feature_objects, test_feature_objects, name_to_category, list(categories), [24], train_vs_test=True)
import IPython; IPython.embed()

