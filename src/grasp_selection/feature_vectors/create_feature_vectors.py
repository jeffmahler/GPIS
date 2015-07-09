import os
import os.path as path
import yaml
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import caffe
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import hstack
from time import time
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

config = yaml.load(open('config/config.yaml', 'r'))
caffe_root = config['caffe_root']
caffe_mode = config['caffe_mode']

batch_size = 100

image_type = 'color'
layer = 'fc7'

path_to_data_dir = 'data/'
path_to_image_dir = image_type+'_images/'
model = path_to_data_dir+'deploy_'+layer+'.prototxt'
pretrained = 'caffe_trials/10/caffenet_train_iter_300000.caffemodel'


class FeatureObject:
	def __init__(self, name, feature_vector):
		self.name = name
		self.feature_vector = feature_vector

class RenderedObject:
	def __init__(self, name, images):
		self.name = name
		self.images = images

def init_cnn():
	caffe.set_mode_gpu() if caffe_mode == 'gpu' else caffe.set_mode_cpu()
	net = caffe.Classifier(model, pretrained,
		mean=np.load(path_to_data_dir+'mean.npy').mean(1).mean(1),
		channel_swap=(2,1,0),
		raw_scale=255,
		image_dims=(256, 256))
	return net

def forward_pass(images, net):
	loaded_images = map(caffe.io.load_image, images)
	final_blobs = map(sp_mat, net.predict(loaded_images, oversample=False))
	return final_blobs

def filter_object_mismatch(rendered_object, images_per_object):
	if len(rendered_object.images) is images_per_object:
		return True
	else:
		print 'MISMATCH ERROR: '+rendered_object.name+' has '+str(len(rendered_object.images))+' images and will be thrown out'
		return False

def feature_objects_from_rendered_objects(rendered_objects, net):
	images_per_object = len(rendered_objects[0].images)
	print 'images per object: '+str(images_per_object)
	rendered_objects = filter(lambda x: filter_object_mismatch(x, images_per_object), rendered_objects)

	print 'Batching forward passes...'
	image_batch = []
	final_blobs = []
	for i, rendered_object in enumerate(rendered_objects):
		if i%int(len(rendered_objects)/20) is 0:
			print str(int((i+1)*100.0/len(rendered_objects)))+'% done'
		image_batch.extend(rendered_object.images)
		while len(image_batch) >= batch_size:
			final_blobs.extend(forward_pass(image_batch[:batch_size], net))
			image_batch = image_batch[batch_size:]
	final_blobs.extend(forward_pass(image_batch[:batch_size], net))
	print 'Finished forward passes'

	feature_objects = []
	index = 0
	for rendered_object in rendered_objects:
		num_images = len(rendered_object.images)
		feature_objects.append(FeatureObject(rendered_object.name, hstack(final_blobs[index:index+num_images])))
		index += num_images
	return feature_objects

def create_feature_objects_from_image_dir(path_to_image_dir, net):
	rendered_object_from_dir = lambda dir: RenderedObject(path.relpath(dir, path_to_image_dir), glob(dir+"*.jpg"))
	rendered_objects = map(rendered_object_from_dir, glob(path_to_image_dir+'*/'))
	return feature_objects_from_rendered_objects(rendered_objects, net)

net = init_cnn()
feature_objects = create_feature_objects_from_image_dir(path_to_image_dir, net)
print 'Feature Vector Size: '+str(feature_objects[0].feature_vector.shape[1])
print 'Saving feature objects to disk...'
pickle.dump(feature_objects, open(path_to_data_dir+'feature_objects_'+image_type+'_'+layer+'.p', "wb"), protocol=2)
print 'Done'

