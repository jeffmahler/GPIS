"""
Feature extractor class for caffe MV-CNNs
Author: Jeff Mahler and Mel Roderick
"""
import database as db
import experiment_config as ec
import features as feat
import feature_extractor as fex

import caffe
import glob
import IPython
import logging
import numpy as np
import os
from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import hstack
from scipy.sparse import vstack
import sys
import time

CHANNEL_SWAP = (2, 1, 0)

class RenderedObject:
    """ Struct for rendered objects """
    def __init__(self, key, images):
        self.key = key
        self.images = images

class MVCNNBatchFeatureExtractor(fex.FeatureExtractor):
    # TODO: update to use database at some point
    def __init__(self, config):
        self.config_ = config
        self.caffe_config_ = self.config_['caffe']
        self._parse_config()
        self.net_ = self._init_cnn()

    def _parse_config(self):
        self.pooling_method_ = self.caffe_config_['pooling_method']
        self.rendered_image_ext_ = self.caffe_config_['rendered_image_ext']
        self.images_per_object_ = self.caffe_config_['images_per_object']
        self.path_to_image_dir_ = self.caffe_config_['rendered_image_dir']
        self.caffe_data_dir_ = self.caffe_config_['config_dir']
        self.batch_size_ = self.caffe_config_['batch_size']
        self.caffe_model_ = os.path.join(self.caffe_data_dir_, self.caffe_config_['caffe_model'])
        self.deploy_model_ = os.path.join(self.caffe_data_dir_, self.caffe_config_['deploy_file']) 
        self.mean_ = np.load(os.path.join(self.caffe_data_dir_, self.caffe_config_['mean_file'])).mean(1).mean(1)

    def _init_cnn(self):
        caffe.set_mode_gpu() if self.caffe_config_['deploy_mode'] == 'gpu' else caffe.set_mode_cpu()
        net = caffe.Classifier(self.deploy_model_, self.caffe_model_,
                               mean=self.mean_,
                               channel_swap=CHANNEL_SWAP,
                               raw_scale=self.caffe_config_['raw_scale'],
                               image_dims=(self.caffe_config_['image_dim_x'], self.caffe_config_['image_dim_x']))
        return net

    def _forward_pass(self, images):
        load_start = time.time()
        loaded_images = map(caffe.io.load_image, images)
        load_stop = time.time()
        logging.debug('Loading took %f sec' %(load_stop - load_start))
        final_blobs = map(sp_mat, self.net_.predict(loaded_images, oversample=False))
        fp_stop = time.time()
        logging.debug('Prediction took %f sec per image' %((fp_stop - load_stop) / len(loaded_images)))
        return final_blobs

    def _filter_object_mismatch(self, rendered_object, images_per_object):
        if len(rendered_object.images) is images_per_object:
            return True
        else:
            logging.error('Mismatch: %s has %d images and will be thrown out' %(rendered_object.key, len(rendered_object.images)))
            return False
        
    def _max_pool(self, vector_list_sparse):
        return reduce(sp_mat.maximum, vector_list_sparse)
        
    def _mean_pool(self, vector_list_sparse):
        num = len(vector_list_sparse)
        return 1.0 / num * reduce(np.add, vector_list_sparse)

    def _feature_vector_for_blobs(self, final_blobs):
        """ Creates a feature vector from the output of a set of featureized images """
        if self.pooling_method_ == 'mean':
            feature_vector = self._mean_pool(final_blobs)
        elif self.pooling_method_ == 'max':
            feature_vector = self._max_pool(final_blobs)
        elif self.pooling_method_ == 'concatenate':
            feature_vector = hstack(final_blobs)
        return feature_vector.toarray().flatten()
            
    def _flush_batch(self, image_batch, rendered_objects, fv_list):
        """ Flushes a batch of images through the network to create a dictionary """
        forward_start = time.time()
        final_blobs = self._forward_pass(image_batch)
        forward_stop = time.time()
        logging.debug('Forward pass took %f sec' %(forward_stop - forward_start))

        for rendered_object in rendered_objects:
            feature_vector = self._feature_vector_for_blobs(final_blobs[:len(rendered_object.images)])
            final_blobs = final_blobs[len(rendered_object.images):]
            fv_list.append(feat.MVCNNFeature(rendered_object.key, feature_vector))

        fv_stop = time.time()
        logging.debug('Postprocessing took %f sec' %(fv_stop - forward_stop))
        
    def _batch_create_feature_vectors(self, rendered_objects):
        """ Creates feature vectors in per-object batches """
        fv_list = []
        image_batch = []
        rendered_object_batch = []
        batch_start = time.time()
        for i, rendered_object in enumerate(rendered_objects):
            image_batch.extend(rendered_object.images)
            rendered_object_batch.append(rendered_object)
            if len(rendered_object_batch) >= self.batch_size_:
                batch_end = time.time()
                logging.debug('Loading took %f sec' %(batch_end - batch_start))
                logging.info('Extracting MV-CNN features for object %d of %d' %(i, len(rendered_objects)))                
                self._flush_batch(image_batch, rendered_object_batch, fv_list)
                batch_start = time.time()
                logging.debug('Batching took %f sec' %(batch_start - batch_end))
                image_batch = []
                rendered_object_batch = []
        if len(image_batch) > 0:
            self._flush_batch(image_batch, rendered_object_batch, fv_list)
        return fv_list

    def _load_rendered_objects(self, graspables):
        """ Load renderable objects for a list of graspables """
        graspable_keys = [g.key for g in graspables]
        if os.path.exists(self.path_to_image_dir_):
            rendered_objects = []
            for key in os.listdir(self.path_to_image_dir_):
                if key in graspable_keys:
                    rendered_images = glob.glob(os.path.join(self.path_to_image_dir_, key, '*%s' %(self.rendered_image_ext_)))
                    rendered_objects.append(RenderedObject(key, rendered_images))
                    rendered_objects = filter(lambda x: self._filter_object_mismatch(x, self.images_per_object_), rendered_objects)
            return rendered_objects
        else:
            logging.warning('Rendered images directory not found: %s' %(self.path_to_image_dir_))
            return None

    def extract(self, graspables):
        # load and batch process the feature vectors
        rendered_objects = self._load_rendered_objects(graspables)
        return self._batch_create_feature_vectors(rendered_objects)
class MVCNNSubsetBatchFeatureExtractor(MVCNNBatchFeatureExtractor):
    def __init__(self, num_images, config):
        self.num_images_ = num_images
        MVCNNBatchFeatureExtractor.__init__(self, config)

    def _feature_vector_for_blobs(self, final_blobs):
        indices = np.random.choice(len(final_blobs), size=self.num_images_, replace=False)
        blob_subset = [final_blobs[i] for i in indices.tolist()]
        return MVCNNBatchFeatureExtractor._feature_vector_for_blobs(self, blob_subset)

class MVCNNImageBatchFeatureExtractor(MVCNNBatchFeatureExtractor):
    def __init__(self, config):
        MVCNNBatchFeatureExtractor.__init__(self, config)

    def _feature_vector_for_blobs(self, final_blobs):
        return final_blobs

    def _flush_batch(self, image_batch, rendered_objects, fv_list):
        """ Flushes a batch of images through the network to create a dictionary """
        forward_start = time.time()
        final_blobs = self._forward_pass(image_batch)
        forward_stop = time.time()
        logging.debug('Forward pass took %f sec' %(forward_stop - forward_start))

        for rendered_object in rendered_objects:
            feature_vectors = self._feature_vector_for_blobs(final_blobs[:len(rendered_object.images)])
            final_blobs = final_blobs[len(rendered_object.images):]
            for feature_vector in feature_vectors:
                fv_list.append(feat.MVCNNFeature(rendered_object.key, feature_vector))

        fv_stop = time.time()
        logging.debug('Postprocessing took %f sec' %(fv_stop - forward_stop))

def test_mvcnn_feature_extractor():
    config_filename = 'cfg/test_mvcnn.yaml'
    config = ec.ExperimentConfig(config_filename)

    chunk = db.FilesystemChunk(config)
    graspables = [g for g in chunk]

    feature_extractor = MVCNNBatchFeatureExtractor(config)
    feature_dict = feature_extractor.extract(graspables)
    IPython.embed()

def test_mvcnn_subset_feature_extractor():
    config_filename = 'cfg/test_mvcnn.yaml'
    config = ec.ExperimentConfig(config_filename)

    chunk = db.FilesystemChunk(config)
    graspables = [g for g in chunk]

    feature_extractor = MVCNNSubsetBatchFeatureExtractor(10, config)
    feature_dict = feature_extractor.extract(graspables)
    IPython.embed()

if __name__ == '__main__':
    test_mvcnn_subset_feature_extractor()
