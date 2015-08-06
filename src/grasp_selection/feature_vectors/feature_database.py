from glob import glob
import os
import yaml
import pickle
import mesh_database
import h5py

from rendered_object import RenderedObject

DATABASE_DIR_KEY = 'database_dir'
MESH_DATABASE_ID_KEY = 'mesh_database_id'
MESH_DATABASE_INDEX_KEY = 'mesh_database_index'
PORTION_TRAINING_KEY = 'portion_training'

RENDERED_IMAGES_KEY = 'rendered_images'
FEATURE_VECTORS_KEY = 'feature_vectors'
NEAREST_FEATURES_KEY = 'nearest_features'
DATASET_SORTER_KEY = 'dataset_sorter'

CAT50_DATASET_KEY = 'CAT50'
SHREC_DATASET_KEY = 'SHREC'

class FeatureDatabase:
	def __init__(self, config=None):
		if config == None:
			config = yaml.load(open('cfg/feature_vectors_config.yaml', 'r'))
		self._parse_config(config)

	def _parse_config(self, config):
		self.database_root_dir_ = config[DATABASE_DIR_KEY]
		self.path_to_image_dir_ = os.path.join(self.database_root_dir_, RENDERED_IMAGES_KEY)
		self.portion_training_ = config[PORTION_TRAINING_KEY]
		self.create_mesh_database(config)

	def rendered_objects(self):
		if os.path.exists(self.path_to_image_dir_):
			rendered_object_from_dir = lambda dir: RenderedObject(str(os.path.relpath(dir, self.path_to_image_dir_)), glob(os.path.join(dir, "*.jpg")))
			rendered_objects = map(rendered_object_from_dir, glob(os.path.join(self.path_to_image_dir_, '*/')))
			return rendered_objects
		else:
			print 'WARNING: Rendered images directory not found: '+self.path_to_image_dir_
			return None

	def portion_training(self):
		return self.portion_training_

	def mesh_database(self):
		return self.mesh_database_

	def create_mesh_database(self, config):
		mesh_database_id = config[MESH_DATABASE_ID_KEY]
		if mesh_database_id == CAT50_DATASET_KEY:
			self.mesh_database_ = mesh_database.Cat50ObjectDatabase(config[MESH_DATABASE_INDEX_KEY])
		elif mesh_database_id == SHREC_DATASET_KEY:
			self.mesh_database_ = mesh_database.SHRECObjectDatabase(config[MESH_DATABASE_INDEX_KEY])
		else:
			print 'Warning: no mesh database matched id in config'

	def feature_vectors(self):
		fv_db_path = os.path.join(self.database_root_dir_, FEATURE_VECTORS_KEY+'.hdf5')
		return h5py.File(fv_db_path, 'r')

	def train_feature_vectors(self):
		train_object_ids = self.feature_dataset_sorter().train_object_keys()
		return dict((k, v) for k, v in self.feature_vectors().items() if k in train_object_ids)

	def test_feature_vectors(self):
		test_object_keys = self.feature_dataset_sorter().test_object_keys()
		return dict((k, v) for k, v in self.feature_vectors().items() if k in train_object_ids)

	def nearest_features(self):
		pca_components = self.load_with_name(NEAREST_FEATURES_KEY+'_pca_components')
		svd = self.load_with_name(NEAREST_FEATURES_KEY+'_svd')
		neighbor_tree = None # self.load_with_name(NEAREST_FEATURES_KEY+'_neighbors_tree')
		import nearest_features as nf
		return nf.NearestFeatures(self, pca_components=pca_components, svd=svd, neighbor_tree=neighbor_tree)

	def save_nearest_features(self, x):
		self.save_data(x.pca_components, NEAREST_FEATURES_KEY+'_pca_components')
		self.save_data(x.svd, NEAREST_FEATURES_KEY+'_svd')
		self.save_data(x.neighbors.tree_, NEAREST_FEATURES_KEY+'_neighbors_tree')

	def feature_dataset_sorter(self):
		return self.load_with_name(DATASET_SORTER_KEY)

	def create_feature_vectors_file(self):
		fv_db_path = os.path.join(self.database_root_dir_, FEATURE_VECTORS_KEY+'.hdf5')
		return h5py.File(fv_db_path, 'w')

	def save_dataset_sorter(self, x):
		self.save_data(x, DATASET_SORTER_KEY)

	def load_with_name(self, name):
		file_path = os.path.join(self.database_root_dir_, name+'.p')
		if os.path.exists(file_path):
			return pickle.load(open(file_path, 'rb'))
		else:
			print 'WARNING: file \''+name+'\' not found'
			return None

	def save_data(self, x, name):
		pickle.dump(x, open(os.path.join(self.database_root_dir_, name+'.p'), 'wb'), protocol=2)

