"""
Feature extractor class for images using caffe CNNs
Author: Jeff Mahler
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
CHANNELS = 3

class CNNBatchFeatureExtractor(fex.FeatureExtractor):
    """ Extract feature descriptors for images in a giant batch """
    def __init__(self, config):
        self.config_ = config
        self.caffe_config_ = self.config_['caffe']
        self._parse_config()
        self.net_ = self._init_cnn()

    def _parse_config(self):
        """ Read the config file """
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
        """ Init the caffe CNN """
        caffe.set_mode_gpu() if self.caffe_config_['deploy_mode'] == 'gpu' else caffe.set_mode_cpu()
        net = caffe.Classifier(self.deploy_model_, self.caffe_model_,
                               mean=self.mean_,
                               channel_swap=CHANNEL_SWAP,
                               raw_scale=self.caffe_config_['raw_scale'],
                               image_dims=(self.caffe_config_['image_dim_x'], self.caffe_config_['image_dim_x']))
        return net

    def _forward_pass(self, images):
        """ Forward pass images through the CNN """
        fp_start = time.time()
        final_blobs = self.net_.predict(images, oversample=False)
        fp_stop = time.time()
        logging.debug('Prediction took %f sec per image' %((fp_stop - fp_start) / len(images)))
        return final_blobs.reshape(final_blobs.shape[0], -1)

    def extract(self, images):
        """ Form feature descriptors for a set of images """
        return self._forward_pass(images)
