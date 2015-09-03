import os
import copy
import random

from feature_database import FeatureDatabase
from nearest_features import NearestFeatures

import sys
sys.path.insert(0, 'src/grasp_selection/')
import experiment_config as ec

index_path = '/Users/MelRod/myProjects/GPIS_data/index_files_test/all_keys.db'
dest_dir = '/Users/MelRod/myProjects/GPIS_data/index_files_test/'
query_keys = ['Cat50_ModelDatabase_bunker_boots', 'BigBIRD_detergent', 'KIT_MelforBottle_800_tex']
dataset_sizes = [10, 100, 1000, 'all']


def create_index_file(keys, filename):
	with open(os.path.join(dest_dir, filename+'.db'), 'w') as index_file:
		for key in keys:
			index_file.write('%s %s\n' % (key, key))

def create_dataset(keys, dataset_name, portion_training=0.7):
	create_index_file(keys, dataset_name)

	randomized_keys = copy.copy(keys)
	random.shuffle(randomized_keys)

	cutoff = int(portion_training*len(randomized_keys))
	create_index_file(randomized_keys[:cutoff], dataset_name+'_train')
	create_index_file(randomized_keys[cutoff:], dataset_name+'_val')

def create_nn_with_keys(all_feature_vectors, keys, suffix):
	feature_vectors = {}
	for key in keys:
		feature_vectors[key] = all_feature_vectors[key]

	nearest_features = NearestFeatures(feature_db, pca_components=100, feature_vectors=feature_vectors)
	feature_db.save_nearest_features(nearest_features, name='nearest_features_'+suffix)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
config = ec.ExperimentConfig(args.config)

all_keys = []
index_file = open(index_path, 'r')
for line in index_file:
	key = line.split()[0]
	if key in query_keys:
		continue
	all_keys.append(key)

all_randomized_keys = copy.copy(all_keys)
random.shuffle(all_randomized_keys)

feature_db = FeatureDatabase(config)
all_feature_vectors = feature_db.feature_vectors()

datasets = []
for size in dataset_sizes:
	if isinstance(size, (int)):
		datasets.append(all_randomized_keys[:size])
	else:
		datasets.append(all_randomized_keys)

for dataset, size in zip(datasets, dataset_sizes):
	suffix = str(size)
	create_dataset(dataset, 'dataset_%s' % (str(size)))
	create_nn_with_keys(all_feature_vectors, dataset, str(size))

