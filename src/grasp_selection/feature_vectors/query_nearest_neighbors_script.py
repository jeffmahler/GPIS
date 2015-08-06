from feature_database import FeatureDatabase
from nearest_features import NearestFeatures

def create_nf(feature_db, query_keys, query_vectors, ext):
	nf = feature_db.nearest_features(name='nearest_features_'+ext)
	for key, vect in zip(query_keys, query_vectors):
		save_query_nf(vect, nf, key+'_'+ext)

def save_query_nf(query_vector, nearest_features, filename):
	nearest_keys, distances = nearest_features.k_nearest_keys(query_vector, k=5)

	f = open(filename+'.db', 'w')
	for key, distance in zip(nearest_keys, distances):
		f.write('%s %s\n' % (key, distance))
	f.close()

query_keys = ['boot', 'detergent', 'bottle_0034']
feature_db = FeatureDatabase()
feature_vectors = feature_db.feature_vectors()

query_vectors = map(lambda x: feature_vectors[x], query_keys)

create_nf(feature_db, query_keys, query_vectors, '10')
create_nf(feature_db, query_keys, query_vectors, '100')
create_nf(feature_db, query_keys, query_vectors, '1000')
create_nf(feature_db, query_keys, query_vectors, 'all')
