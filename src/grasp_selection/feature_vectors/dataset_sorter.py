import random
import pickle
import object_database

class DatasetSorter:
	def __init__(self, )
	# def __init__(self, object_database, portion_training=0.75):
	# 	object_ids = object_database.object_ids().keys()
	# 	random.shuffle(object_ids)
	# 	length = len(object_ids)
	# 	self.train_object_names = object_ids[:int(length*portion_training)]
	# 	self.test_object_names = object_ids[int(length*portion_training):length]

	def save_to_dir(self, path_to_data_dir):
		pickle.dump(self, open(path_to_data_dir+'dataset_sorter.p', "wb"), protocol=2)
