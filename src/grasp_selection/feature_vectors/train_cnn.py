import os
import random

from feature_database import FeatureDatabase
from rendered_object import RenderedObject

def save_cnn_train_val_sets(train_object_keys, feature_db, caffe_data_dir, portion_training):
	rendered_objects = filter(lambda x: x.key in train_object_keys, feature_db.rendered_objects())
	mesh_db = feature_db.mesh_database()

	id = 0
	category_to_id = {}
	training_set = []
	for rendered_object in rendered_objects:
		category = mesh_db.object_category_for_key(rendered_object.key)
		if not category in category_to_id:
			category_to_id[category] = id
			id += 1
		for image in rendered_object.images:
			training_set.append(image+' '+str(category_to_id[category])+'\n')

	random.shuffle(training_set)

	length = len(training_set)
	train_set_path = caffe_data_dir+'train.txt'
	val_set_path = caffe_data_dir+'val.txt'

	if not os.path.exists(train_set_path):
		train = open(train_set_path, 'w')
		for i in range(0, int(length*portion_training)):
			train.write(training_set[i])
		train.close()
	
	if not os.path.exists(val_set_path):
		val = open(val_set_path, 'w')
		for i in range(int(length*portion_training), length):
			val.write(training_set[i])
		val.close()

def train(feature_db, caffe_config, dataset_sorter):
	save_cnn_train_val_sets(dataset_sorter.train_object_keys(), feature_db, caffe_config.caffe_data_dir(), caffe_config.portion_training())
	os.chdir(caffe_config.caffe_data_dir())
	os.system(caffe_config.caffe_tools()+'caffe train -solver '+caffe_config.solver()+' -weights '+caffe_config.finetuning_model())
