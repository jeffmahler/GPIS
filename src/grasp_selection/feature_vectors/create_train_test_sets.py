from glob import glob
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path_to_index',
					help='the path to the index file denoting the category of each object file')
parser.add_argument('path_to_data_dir',
					help='the path to the dir to store training data')
parser.add_argument('path_to_image_dir',
					help='the path to the dir where the training images are stored')
parser.add_argument('--portion_training', type=float, default=0.75,
					help='the portion of objects used for the training set')
parser.add_argument('--portion_cnn_training', type=float, default=0.75,
					help='the portion of the images used from the training set objects used for the cnn training set')

args = parser.parse_args()
path_to_index = args.path_to_index
path_to_data_dir = args.path_to_data_dir
path_to_image_dir = args.path_to_image_dir
portion_training = args.portion_training
portion_cnn_training = args.portion_cnn_training


def create_training_test_objects(path_to_image_dir):
	os.chdir(path_to_image_dir)
	object_names = map(lambda dir: dir[:-1], glob("*/"))
	random.shuffle(object_names)
	length = len(object_names)
	training_objects = object_names[:int(length*portion_training)]
	test_objects = object_names[int(length*portion_training):length]
	return training_objects, test_objects

def create_categories(path_to_index):
	categories = {}
	with open(path_to_index) as file:
		for line in file:
			split = line.split()
			file_name = split[0]
			category = split[1]
			if category in categories:
				categories[category].append(file_name)
			else:
				categories[category] = [file_name]
	return categories

def save_training_test_sets(training_objects, test_objects):
	training_objects_file = open(path_to_data_dir+'training_objects.txt', 'w')
	test_objects_file = open(path_to_data_dir+'test_objects.txt', 'w')
	map(lambda name: training_objects_file.write(name+'\n'), training_objects)
	map(lambda name: test_objects_file.write(name+'\n'), test_objects)
	training_objects_file.close()
	test_objects_file.close()

def save_cnn_train_val_sets(training_objects, categories):
	category_to_id = open(path_to_data_dir+'category_to_id.txt', 'w')
	training_set = []
	id = 0
	for category in categories:
		category_to_id.write(category+' '+str(id)+'\n')
		for object_name in categories[category]:
			directory = path_to_image_dir+object_name+'/'
			if object_name in training_objects and os.path.isdir(directory):
				os.chdir(directory)
				for image_file in glob('*.jpg'):
					training_set.append(directory+image_file+' '+str(id)+'\n')
		id += 1
	category_to_id.close()

	random.shuffle(training_set)

	length = len(training_set)
	train = open(path_to_data_dir+'train.txt', 'w')
	val = open(path_to_data_dir+'val.txt', 'w')
	for i in range(0, int(length*portion_training)):
		train.write(training_set[i])
	for i in range(int(length*portion_training), length):
		val.write(training_set[i])
	train.close()
	val.close()

training_objects, test_objects = create_training_test_objects(path_to_image_dir)
save_training_test_sets(training_objects, test_objects)
categories = create_categories(path_to_index)
save_cnn_train_val_sets(training_objects, categories)
