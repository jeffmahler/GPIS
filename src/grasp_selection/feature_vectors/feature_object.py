import os
import os.path as path
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
import paths
import time
from render_object import RenderedObject


class FeatureObject:
	def __init__(self, key, feature_vector):
		self.key = key
		self.feature_vector = feature_vector

class FeatureObjectDatabase:
	def __init__(path_to_image_dir, deploy_model, caffemodel, batch_size=1000, max_pooling=True, caffe_mode='cpu'):
		self.path_to_image_dir = path_to_image_dir
		self.deploy_model = deploy_model
		self.caffemodel = caffemodel

		net = init_cnn(caffemodel, deploy_model, caffe_mode)
		self.feature_objects_ = create_feature_objects_from_image_dir(path_to_image_dir, net, batch_size, max_pooling)

	def get_feature_object_list(self):
		return self.feature_objects_.values()

	def get_feature_object_dict(self):
		return self.feature_objects_

	def save_to(self, save_dir):
		pickle.dump(feature_objects, open(save_dir+'feature_objects.p', "wb"), protocol=2)



	def init_cnn(caffemodel, deploy_model, caffe_mode):
		caffe.set_mode_gpu() if caffe_mode == 'gpu' else caffe.set_mode_cpu()
		net = caffe.Classifier(deploy_model, caffemodel,
			mean=np.load(paths.mean_file).mean(1).mean(1),
			channel_swap=(2,1,0),
			raw_scale=255,
			image_dims=(256, 256))
		return net

	def forward_pass(images, net):
		loaded_images = map(caffe.io.load_image, images)
		final_blobs = map(sp_mat, net.predict(loaded_images, oversample=False))
		return final_blobs

	def batch_forward_passes(rendered_objects, net, batch_size):
		image_batch = []
		final_blobs = []
		for i, rendered_object in enumerate(rendered_objects):
			if i%int(len(rendered_objects)/20+1) is 0:
				print str(int((i+1)*100.0/len(rendered_objects)))+'% done'
			image_batch.extend(rendered_object.images)
			while len(image_batch) >= batch_size:
				final_blobs.extend(forward_pass(image_batch[:batch_size], net))
				image_batch = image_batch[batch_size:]
		if len(image_batch) > 0:
			final_blobs.extend(forward_pass(image_batch, net))
		return final_blobs

	def filter_object_mismatch(rendered_object, images_per_object):
		if len(rendered_object.images) is images_per_object:
			return True
		else:
			print 'MISMATCH ERROR: '+rendered_object.key+' has '+str(len(rendered_object.images))+' images and will be thrown out'
			return False

	def max_pool(vector_list_sparse):
		return reduce(sp_mat.maximum, vector_list_sparse)

	def mean_pool(vector_list_sparse):
		num = len(vector_list_sparse)
		return np.multiply(1.0/num, reduce(np.add, vector_list_sparse))

	def create_feature_objects_with_blobs(final_blobs, rendered_objects, max_pooling):
		feature_objects = {}
		index = 0
		for i, rendered_object in enumerate(rendered_objects):
			if i%int(len(rendered_objects)/20+1) is 0:
				print str(int((i+1)*100.0/len(rendered_objects)))+'% done'
			num_images = len(rendered_object.images)
			blobs_in_range = final_blobs[index:index+num_images]
			if True:
				feature_vector = mean_pool(blobs_in_range)
			elif max_pooling:
				feature_vector = max_pool(blobs_in_range)
			else:
				feature_vector = hstack(blobs_in_range)
			feature_objects[rendered_object.key] = FeatureObject(rendered_object.key, feature_vector)
			index += num_images
		return feature_objects

	def feature_objects_from_rendered_objects(rendered_objects, net, batch_size, max_pooling):
		images_per_object = len(rendered_objects[0].images)
		print 'images per object: '+str(images_per_object)
		rendered_objects = filter(lambda x: filter_object_mismatch(x, images_per_object), rendered_objects)

		start_time = time.time()
		print 'Batching forward passes...'
		final_blobs = batch_forward_passes(rendered_objects, net, batch_size)
		print 'Finished forward passes: '+str(time.time()-start_time)+'s'

		start_time = time.time()
		print 'Creating feature objects...'
		feature_objects = create_feature_objects_with_blobs(final_blobs, rendered_objects, max_pooling)
		print 'Finished creating feature objects: '+str(time.time()-start_time)+'s'
		return feature_objects

	def create_feature_objects_from_image_dir(path_to_image_dir, net, batch_size, max_pooling):
		rendered_object_from_dir = lambda dir: RenderedObject(path.relpath(dir, path_to_image_dir), glob(dir+"*.jpg"))
		rendered_objects = map(rendered_object_from_dir, glob(path_to_image_dir+'*/')[:2])
		return feature_objects_from_rendered_objects(rendered_objects, net, batch_size, max_pooling)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("path_to_image_dir",
						help="the path to the image directory")
	parser.add_argument("deploy_model",
						help="the path to the deploy model")
	parser.add_argument("caffemodel",
						help="the path to the caffemodel")
	parser.add_argument("--batch_size", type=int, default=1000,
						help="the directory to save in")
	parser.add_argument("--save_dir", default=None,
						help="the directory to save in")
	parser.add_argument("-m", "--max_pool", action="store_true",
						help="max pool feature_vecotrs")
	args = parser.parse_args()
	feature_object_db = FeatureObjectDatabase(args.path_to_image_dir, args.deploy_model, args.caffemodel, max_pool=args.max_pool)

	save_dir = args.save_dir
	if save_dir != None:
		feature_object_db.save_to(save_dir)



