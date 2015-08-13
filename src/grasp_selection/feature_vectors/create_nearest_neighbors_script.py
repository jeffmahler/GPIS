import copy
import random

from feature_database import FeatureDatabase
from nearest_features import NearestFeatures

def create_save_nn(all_feature_vectors, random_keys, num_items=10):
	feature_vectors = {}
	for i in range(0, num_items):
		key = random_keys[i]
		feature_vectors[key] = all_feature_vectors[key]

	nearest_features = NearestFeatures(feature_db, pca_components=100, feature_vectors=feature_vectors)
	feature_db.save_nearest_features(nearest_features, name='nearest_features_'+str(num_items))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
config = ec.ExperimentConfig(args.config)

feature_db = FeatureDatabase(config)

query_keys = ['Cat50_ModelDatabase_bunker_boots', 'BigBIRD_detergent', 'KIT_MelforBottle_800_tex'] # 'KIT_Waterglass_800_tex']
feature_vectors_db = feature_db.feature_vectors()
feature_vectors = {}
for key in feature_vectors_db.keys():
	if key in query_keys:
		continue
	feature_vectors[key] = feature_vectors_db[key]

random_keys = copy.copy(feature_vectors.keys())
random.shuffle(random_keys)

create_save_nn(feature_vectors, random_keys, num_items=15)
create_save_nn(feature_vectors, random_keys, num_items=150)

nearest_features = NearestFeatures(feature_db, pca_components=100, feature_vectors=feature_vectors)
feature_db.save_nearest_features(nearest_features, name='nearest_features_all')
