import caffe
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import hstack
from time import time
from nearpy.hashes import RandomBinaryProjections
import time
import feature_database
import caffe_config
import numpy as np
import h5py

class FeatureVectorDatabase:
	def __init__(self, feature_db, caffe_config):
		net = self._init_cnn(caffe_config)
		self.batch_size = caffe_config.deploy_batch_size()
		self.vector_merging_method = caffe_config.vector_merging_method()
		self.feature_db = feature_db
		self.feature_objects_ = self._create_feature_vectors(net)

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

	def _feature_vector_for_blobs(self, final_blobs):
		if self.vector_merging_method == 'mean':
			feature_vector = self._mean_pool(final_blobs)
		elif self.vector_merging_method == 'max':
			feature_vector = self._max_pool(final_blobs)
		else:
			feature_vector = hstack(final_blobs)
		return feature_vector.toarray().flatten()

	def _flush_batch(self, image_batch, rendered_objects, net, fv_dict):
		final_blobs = []
		while len(image_batch) > 0:
			final_blobs.extend(self._forward_pass(image_batch[:self.batch_size], net))
			image_batch = image_batch[self.batch_size:]

		for rendered_object in rendered_objects:
			feature_vector = self._feature_vector_for_blobs(final_blobs[:len(rendered_object.images)])
			final_blobs = final_blobs[len(rendered_object.images):]
			fv_dict[rendered_object.key] = feature_vector

	def _batch_create_feature_vectors(self, rendered_objects, net, fv_dict):
		image_batch = []
		final_blobs = []
		rendered_object_batch = []
		for i, rendered_object in enumerate(rendered_objects):
			image_batch.extend(rendered_object.images)
			rendered_object_batch.append(rendered_object)
			if len(image_batch) >= self.batch_size:
				self._flush_batch(image_batch, rendered_object_batch, net, fv_dict)
				image_batch = []
				rendered_object_batch = []
				print str(int((i+1)*100.0/len(rendered_objects)))+'%'
		if len(image_batch) > 0:
			self._flush_batch(image_batch, rendered_object_batch, net, fv_dict)
		return final_blobs

	def _create_feature_vectors(self, net):
		# rendered_objects = self.feature_db.rendered_objects()
		# images_per_object = len(rendered_objects[0].images)
		# print 'images per object: '+str(images_per_object)
		# rendered_objects = filter(lambda x: self._filter_object_mismatch(x, images_per_object), rendered_objects)

		# start_time = time.time()
		# print 'Creating feature vectors...'
		# fv_dict = self.feature_db.create_feature_vectors_file(overwrite=True)
		# final_blobs = self._batch_create_feature_vectors(rendered_objects, net, fv_dict)
		# print 'Finished creating feature vectors: '+str(time.time()-start_time)+'s'
		import os
		fv_db_path = os.path.join(feature_db.database_root_dir_, 'feature_vectors.hdf5')
		fv_dict = h5py.File(fv_db_path, 'r+')

		for key in fv_dict.keys():
			if 'Orig_tex' in key:
				fv_dict[key.replace('Orig', '800')] = fv_dict[key]
				fv_dict.__delitem__(key)

		import IPython; IPython.embed()

if __name__ == '__main__':
	feature_db = feature_database.FeatureDatabase()
	caffe_config = caffe_config.CaffeConfig()
	feature_object_db = FeatureVectorDatabase(feature_db, caffe_config)


