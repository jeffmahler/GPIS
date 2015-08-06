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

feature_db = FeatureDatabase()

query_keys = ['boot', 'detergent', 'bottle_0034']
feature_vectors_db = feature_db.feature_vectors()
feature_vectors = {}
for key in feature_vectors_db.keys():
	if key in query_keys:
		continue
	feature_vectors[key] = feature_vectors_db[key]

random_keys = copy.copy(all_feature_vectors.keys())
random.shuffle(random_keys)

create_save_nn(feature_vectors, random_keys, num_items=10)
create_save_nn(feature_vectors, random_keys, num_items=100)
create_save_nn(feature_vectors, random_keys, num_items=1000)

nearest_features = NearestFeatures(feature_db, pca_components=100, feature_vectors=feature_vectors)
feature_db.save_nearest_features(nearest_features, name='nearest_features_all')
