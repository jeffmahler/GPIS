import os
from glob import glob
import random

import paths

def save_cnn_train_val_sets(train_object_ids, categories, path_to_image_dir, portion_training):
	category_to_id = open(paths.path_to_data_dir+'category_to_id.txt', 'w')
	training_set = []
	id = 0
	for category in categories:
		category_to_id.write(category+' '+str(id)+'\n')
		for object_name in categories[category]:
			directory = path_to_image_dir+object_name+'/'
			if object_name in train_object_ids and os.path.isdir(directory):
				os.chdir(directory)
				for image_file in glob('*.jpg'):
					training_set.append(directory+image_file+' '+str(id)+'\n')
		id += 1
	category_to_id.close()

	random.shuffle(training_set)

	length = len(training_set)
	print(os.getcwd())
	train = open(paths.path_to_data_dir+'train.txt', 'w')
	val = open(paths.path_to_data_dir+'val.txt', 'w')
	for i in range(0, int(length*portion_training)):
		train.write(training_set[i])
	for i in range(int(length*portion_training), length):
		val.write(training_set[i])
	train.close()
	val.close()

def train(train_object_ids, object_database, path_to_image_dir, portion_training=0.75):
	save_cnn_train_val_sets(train_object_ids, object_database.categories(), path_to_image_dir, portion_training)
	os.system(paths.caffe_tools+'caffe train -solver '+paths.solver+' -weights '+paths.weights)
