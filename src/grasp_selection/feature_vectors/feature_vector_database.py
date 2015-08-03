import caffe
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import hstack
from time import time
from nearpy.hashes import RandomBinaryProjections
import time
import rendered_object
import feature_database
import caffe_config
import numpy as np
import h5py

class FeatureObjectDatabase:
	def __init__(self, feature_db, caffe_config):
		net = self._init_cnn(caffe_config)
		self.feature_objects_ = self._create_feature_objects_with_database(feature_db, net, caffe_config.deploy_batch_size())

	def _init_cnn(self, caffe_config):
		caffe.set_mode_gpu() if caffe_config.deploy_mode() == 'gpu' else caffe.set_mode_cpu()
		net = caffe.Classifier(caffe_config.deploy_model(), caffe_config.caffe_model(),
			mean=caffe_config.mean(),
			channel_swap=(2,1,0),
			raw_scale=255,
			image_dims=(256, 256))
		return net

	def _forward_pass(self, images, net):
		loaded_images = map(caffe.io.load_image, images)
		final_blobs = map(sp_mat, net.predict(loaded_images, oversample=False))
		return final_blobs

	def _batch_forward_passes(self, rendered_objects, net, batch_size):
		image_batch = []
		final_blobs = []
		for i, rendered_object in enumerate(rendered_objects):
			if i%int(len(rendered_objects)/100+1) is 0:
				print str(int((i+1)*100.0/len(rendered_objects)))+'% done'
			image_batch.extend(rendered_object.images)
			while len(image_batch) >= batch_size:
				final_blobs.extend(self._forward_pass(image_batch[:batch_size], net))
				image_batch = image_batch[batch_size:]
		if len(image_batch) > 0:
			final_blobs.extend(self._forward_pass(image_batch, net))
		return final_blobs

	def _filter_object_mismatch(self, rendered_object, images_per_object):
		if len(rendered_object.images) is images_per_object:
			return True
		else:
			print 'MISMATCH ERROR: '+rendered_object.key+' has '+str(len(rendered_object.images))+' images and will be thrown out'
			return False

	def _max_pool(self, vector_list_sparse):
		return reduce(sp_mat.maximum, vector_list_sparse)

	def _mean_pool(self, vector_list_sparse):
		num = len(vector_list_sparse)
		return np.multiply(1.0/num, reduce(np.add, vector_list_sparse))

	def _create_feature_vectors(self, fv_dict, final_blobs, rendered_objects, vector_merging_method):
		index = 0
		for i, rendered_object in enumerate(rendered_objects):
			if i%int(len(rendered_objects)/20+1) is 0:
				print str(int((i+1)*100.0/len(rendered_objects)))+'% done'
			num_images = len(rendered_object.images)
			blobs_in_range = final_blobs[index:index+num_images]
			if vector_merging_method == 'mean':
				feature_vector = self._mean_pool(blobs_in_range)
			elif vector_merging_method == 'max':
				feature_vector = self._max_pool(blobs_in_range)
			else:
				feature_vector = hstack(blobs_in_range)
			fv_dict[rendered_object.key] = feature_vector.toarray().flatten()
			index += num_images

	def _create_feature_objects_with_database(self, feature_database, net, batch_size):
		rendered_objects = feature_database.rendered_objects()
		images_per_object = len(rendered_objects[0].images)
		print 'images per object: '+str(images_per_object)
		rendered_objects = filter(lambda x: self._filter_object_mismatch(x, images_per_object), rendered_objects)

		start_time = time.time()
		print 'Batching forward passes...'
		final_blobs = self._batch_forward_passes(rendered_objects, net, batch_size)
		print 'Finished forward passes: '+str(time.time()-start_time)+'s'

		start_time = time.time()
		print 'Creating and saving feature objects...'
		fv_dict = feature_database.create_feature_vectors_file()
		self._create_feature_vectors(fv_dict, final_blobs, rendered_objects, caffe_config.vector_merging_method())
		print 'Finished creating feature objects: '+str(time.time()-start_time)+'s'

if __name__ == '__main__':
	feature_db = feature_database.FeatureDatabase()
	caffe_config = caffe_config.CaffeConfig()
	feature_object_db = FeatureObjectDatabase(feature_db, caffe_config)

