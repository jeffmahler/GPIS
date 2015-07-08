import os
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import pickle
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import hstack
from time import time
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from numpy import argmax

gpis_root = '/Users/MelRod/myProjects/GPIS/'
sys.path.insert(0, gpis_root+'src/grasp_selection')
import kernels

caffe_root = '/Users/MelRod/lib/caffe-master/'
dex_net_root = '/Users/MelRod/myProjects/dex_net/'

batch_size = 1000

path_to_data_dir = dex_net_root+'data/'
path_to_image_dir = dex_net_root+'mesh_images/'
model = path_to_data_dir+'deploy.prototxt'
pretrained = dex_net_root+'caffe_trials/7/caffenet_train_iter_30000.caffemodel'


path_to_index_file = path_to_data_dir+'Cat50_ModelDatabase_index.db'
path_to_category_file = path_to_data_dir+'category_to_id.txt'

class ClassifiedObject:
	def __init__(self, name, classifications, category):
		self.name = name
		self.classifications = classifications
		self.category = category

class RenderedObject:
	def __init__(self, name, images):
		self.name = name
		self.images = images

def create_name_to_category(path_to_file):
	name_to_category = {}
	with open(path_to_file) as file:
		for line in file:
			split = line.split()
			file_name = split[0]
			category = split[1]
			name_to_category[file_name] = category
	return name_to_category

def create_category_list(path_to_file):
	name_to_category = {}
	categories = []
	with open(path_to_file) as file:
		for line in file:
			categories.append(line.split()[0])
	return categories

def batch_forward_pass(images):
	caffe.set_mode_cpu()
	net = caffe.Classifier(model, pretrained,
		mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
		channel_swap=(2,1,0),
		raw_scale=255,
		image_dims=(256, 256))
	print 'Loading images...'
	loaded_images = []
	for i in range(0, len(images)):
		loaded_images.append(caffe.io.load_image(images[i]))
		if (i+1)%int(len(images)/20) is 0:
			print str(int((i+1)*100.0/len(images)))+'% done'
	print 'Finished loading images'
	print 'Batching forward passes...'
	final_blobs = []
	for i in range(0, len(loaded_images), batch_size):
		stop = i+batch_size
		if stop > len(loaded_images): stop = len(loaded_images)
		final_blobs.extend(map(argmax, net.predict(loaded_images[i:stop], oversample=False)))
		print str(int((stop+1)*100.0/len(loaded_images)))+'% done'
	print 'Finished forward passes'
	return final_blobs

def classified_objects_from_rendered_objects(rendered_objects):
	images_per_object = len(rendered_objects[0].images)
	print 'images per object: '+str(images_per_object)
	all_images = []
	for i in range(len(rendered_objects)-1, -1, -1):
		rendered_object = rendered_objects[i]
		if len(rendered_object.images) is images_per_object:
			all_images.extend(rendered_object.images)
		else:
			print 'MISMATCH ERROR: '+rendered_object.name+' has '+str(len(rendered_object.images))+' images and will be thrown out'
			rendered_objects.pop(i)
	all_images.reverse()
	classifications = batch_forward_pass(all_images)
	classified_objects = []
	index = 0
	for rendered_object in rendered_objects:
		num_images = len(rendered_object.images)
		classified_objects.append(ClassifiedObject(rendered_object.name, classifications[index:index+num_images], name_to_category[rendered_object.name]))
		index += num_images
	return classified_objects

def create_classified_objects_from_image_dir(path_to_image_dir):
	os.chdir(path_to_image_dir)
	rendered_object_from_dir = lambda dir: RenderedObject(dir[:-1], glob(dir+"*.jpg"))
	rendered_objects = map(rendered_object_from_dir, glob("*/"))
	return classified_objects_from_rendered_objects(rendered_objects)

name_to_category = create_name_to_category(path_to_index_file)
categories = create_category_list(path_to_category_file)
classified_objects = create_classified_objects_from_image_dir(path_to_image_dir)

num_total_correct = 0
num_total = 0
for classified_object in classified_objects:
	num_correct = 0
	for classification in classified_object.classifications:
		pred_category = categories[classification]
		if pred_category == classified_object.category:
			num_correct += 1
	num = len(classified_object.classifications)
	print classified_object.name+': '+str(int(num_correct*100/num))+"% correct"
	num_total_correct += num_correct
	num_total += num
print 'TOTAL ACCURACY: '+str(int(num_total_correct*100/num_total))+"% correct"
# print 'Saving feature objects to disk...'
# pickle.dump(classified_objects, open(path_to_data_dir+'classified_objects_pool5.p', "wb"))
# print 'Done'

