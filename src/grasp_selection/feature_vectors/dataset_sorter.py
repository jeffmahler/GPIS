import random
import mesh_database
from feature_database import FeatureDatabase

class DatasetSorter:
	def __init__(self, feature_db):
		object_keys = feature_db.mesh_database().object_keys()
		random.shuffle(object_keys)
		length = len(object_keys)
		self.train_object_keys_ = object_keys[:int(length*feature_db.portion_training())]
		self.test_object_keys_ = object_keys[int(length*feature_db.portion_training()):length]

	def train_object_keys(self):
		return self.train_object_keys_

	def test_object_keys(self):
		return self.test_object_keys_

