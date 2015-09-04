import os
import copy
import random
from sets import Set
import matplotlib.pyplot as plt
import numpy as np
import argparse
import IPython
import logging
import pickle as pkl

import feature_database as fdb
from nearest_features import NearestFeatures

import sys
sys.path.insert(0, 'src/grasp_selection/')
import experiment_config as ec

def scatter_feature_objects(plt, feature_objects, color, s=50):
	x = map(lambda obj: obj.feature_vector[0], feature_objects)
	y = map(lambda obj: obj.feature_vector[1], feature_objects)
	return plt.scatter(x, y, s=s, c=color)

def plot_feature_objects_data(feature_objects, mesh_database, categories, s=50, num_to_plot=1000):
	all_feature_objects = copy.copy(feature_objects)
	random.shuffle(feature_objects)
	feature_objects = feature_objects[:num_to_plot]

	objs = []
	labels = ['Others']
	plt.figure()
	colors = ['b'] # plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(categories)))
	objs.append(scatter_feature_objects(plt, feature_objects, '#eeeeff', s=s))

	i = 0
	for category in categories:
		filter_for_cat = lambda obj: mesh_database.object_category_for_key(obj.key[13:]) == category
		sub_fobjs = filter(filter_for_cat, all_feature_objects)
		objs.append(scatter_feature_objects(plt, sub_fobjs, colors[i], s=s))
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
		filter_for_cat = lambda obj: name_to_category[obj.key] == category
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
config = ec.ExperimentConfig(args.config)

feature_db = fdb.FeatureDatabase(config)

plot_feature_objects_data(feature_db.nearest_features(name='nearest_features_all').neighbors.data_, feature_db.mesh_database(), ['mug'], s=30)
import IPython; IPython.embed()

