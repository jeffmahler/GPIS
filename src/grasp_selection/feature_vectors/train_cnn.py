import os
from glob import glob
import random

import paths

def save_cnn_train_val_sets(train_object_keys, feature_database, config_dir, portion_training):
	category_to_id = open(paths.path_to_data_dir+'category_to_id.txt', 'w')
	rendered_objects = filter(lambda x: x.key in train_object_keys, feature_database.rendered_objects())
	mesh_db = feature_database.mesh_database()
	
	id = 0
	category_to_id = {}
	for rendered_object in rendered_objects:
		category = mesh_db.object_category_for_id(rendered_object.key)
		if !(category in category_to_id):
			category_to_id[category] = id
			id += 1
		for image in rendered_object.images():
			training_set.append(directory+image_file+' '+str(category_to_id[category])+'\n')

	random.shuffle(training_set)

	length = len(training_set)
	print(os.getcwd())
	train = open(config_dir+'train.txt', 'w')
	val = open(config_dir+'val.txt', 'w')
	for i in range(0, int(length*portion_training)):
		train.write(training_set[i])
	for i in range(int(length*portion_training), length):
		val.write(training_set[i])
	train.close()
	val.close()

def train(feature_database, caffe_config, dataset_sorter):
	save_cnn_train_val_sets(dataset_sorter.train_object_keys(), feature_database.categories(), caffe_config.config_dir(), caffe_config.portion_training())
	os.system(caffe_config.caffe_tools()+'caffe train -solver '+caffe_config.solver()+' -weights '+caffe_config.finetuning_model())
