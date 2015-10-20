import os
import copy
import math
import random
import IPython

from feature_database import FeatureDatabase
from nearest_features import NearestFeatures

import sys
sys.path.insert(0, 'src/grasp_selection/')
import experiment_config as ec

index_path = '/home/jmahler/jeff_working/GPIS/data/index_files/big_set/all_keys.db'
dest_dir = '/home/jmahler/jeff_working/GPIS/data/index_files/big_set/'
query_keys = ['BigBIRD_detergent', 'KIT_MelforBottle_800_tex', 'YCB_black_and_decker_lithium_drill_driver_unboxed', 'Cat50_ModelDatabase_bunker_boots']
#index_path = '/Users/MelRod/myProjects/GPIS_data/index_files_test/all_keys.db'
#dest_dir = '/Users/MelRod/myProjects/GPIS_data/index_files_test/'
#query_keys = ['Cat50_ModelDatabase_bunker_boots', 'BigBIRD_detergent', 'KIT_MelforBottle_800_tex']
training_sizes = [10, 100, 1000, 10000]
val_ratio = 0.3
index_file_template = 'keys_%s'

def create_index_file(keys, filename):
	with open(os.path.join(dest_dir, filename+'.db'), 'w') as index_file:
		for key in keys:
			index_file.write('%s %s\n' % (key, key))

def create_nn_with_keys(all_feature_vectors, keys, suffix):
	feature_vectors = {}
	for key in keys:
                #print key
		feature_vectors[key] = all_feature_vectors[key]

	nearest_features = NearestFeatures(feature_db, pca_components=100, feature_vectors=feature_vectors)
	feature_db.save_nearest_features(nearest_features, name='nearest_features_'+suffix)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
config = ec.ExperimentConfig(args.config)

# get all keys and randomize
all_keys = []
index_file = open(index_path, 'r')
for line in index_file:
	key = line.split()[0]
	if key in query_keys:
		continue
	all_keys.append(key)

all_randomized_keys = copy.copy(all_keys)
random.shuffle(all_randomized_keys)

# split training and test now (test will be the remaining items after taking the largest training set)
all_training_keys = all_randomized_keys[:training_sizes[-1]]
all_test_keys = all_randomized_keys[training_sizes[-1]:]

test_ratio = float(len(all_test_keys)) / float(len(all_training_keys))
test_sizes = [int(test_ratio * s) for s in training_sizes]

val_sizes = [int(val_ratio * s) for s in training_sizes]
training_subset_sizes = [s - v for s,v in zip(training_sizes, val_sizes)]
all_val_keys = all_training_keys[:val_sizes[-1]]
all_training_subset_keys = all_training_keys[val_sizes[-1]:]

# create list of training and test model keys
training_datasets = []
val_datasets = []
test_datasets = []
for train_size, val_size, test_size in zip(training_subset_sizes, val_sizes, test_sizes):
	if isinstance(train_size, (int)):
		training_datasets.append(all_training_subset_keys[:train_size])
	else:
		training_datasets.append(all_training_subset_keys)

	if isinstance(val_size, (int)):
		val_datasets.append(all_val_keys[:val_size])
	else:
		val_datasets.append(all_val_keys)

	if isinstance(test_size, (int)):
		test_datasets.append(all_test_keys[:test_size])
	else:
		test_datasets.append(all_test_keys)

# load all feature vectors
feature_db = FeatureDatabase(config)
all_feature_vectors = feature_db.feature_vectors()

# add training datasets
print
print 'Creating training datasets'
for training_dataset, training_size in zip(training_datasets, training_sizes):
	suffix = str(training_size) + '_train'
        print 'Creating set', suffix
	create_index_file(training_dataset, index_file_template % (suffix))
	create_nn_with_keys(all_feature_vectors, training_dataset, suffix)

# add val datasets
print
print 'Creating validation datasets'
for val_dataset, training_size in zip(val_datasets, training_sizes):
	suffix = str(training_size) + '_val'
        print 'Creating set', suffix
	create_index_file(val_dataset, index_file_template % (suffix))
	create_nn_with_keys(all_feature_vectors, val_dataset, suffix)

# add train+val datasets
print
print 'Creating combined datasets'
for training_dataset, val_dataset, training_size in zip(training_datasets, val_datasets, training_sizes):
        combined_dataset = list(training_dataset)
        combined_dataset.extend(val_dataset)
	suffix = str(training_size) + '_train_and_val'
        print 'Creating set', suffix
	create_index_file(combined_dataset, index_file_template % (suffix))
	create_nn_with_keys(all_feature_vectors, combined_dataset, suffix)

# add test datasets
print
print 'Creating test datasets'
for dataset, training_size in zip(test_datasets, training_sizes):
        test_dataset = list(query_keys) # ensure that the test data contains the query keys
        test_dataset.extend(dataset)
	suffix = str(training_size) + '_test'
        print 'Creating set', suffix
	create_index_file(test_dataset, index_file_template % (suffix))
	create_nn_with_keys(all_feature_vectors, test_dataset, suffix)



