import yaml
import numpy as np
import os.path as path

CAFFE_TOOLS_KEY = 'caffe_tools'
CAFFE_DATA_DIR_KEY = 'config_dir'
PORTION_TRAINING_KEY = 'portion_training'
CAFFE_MODEL_KEY = 'caffe_model'
CAFFE_FINETUNING_MODEL_KEY = 'caffe_finetuning_model'
DEPLOY_MODE_KEY = 'deploy_mode'
BATCH_SIZE_KEY = 'batch_size'
VECTOR_MERGING_METHOD_KEY = 'vector_merging_method'


class CaffeConfig(object):
	def __init__(self, config=None):
		if config == None:
			config = yaml.load(open('cfg/caffe_config.yaml', 'r'))
		self._parse_config(config)

	def _parse_config(self, config):
		self.caffe_tools_ = config[CAFFE_TOOLS_KEY]
		self.caffe_data_dir_ = config[CAFFE_DATA_DIR_KEY]
		self.portion_training_ = config[PORTION_TRAINING_KEY]
		self.caffe_model_ = config[CAFFE_MODEL_KEY]
		self.caffe_finetuning_model_ = config[CAFFE_FINETUNING_MODEL_KEY]
		self.deploy_mode_ = config[DEPLOY_MODE_KEY]
		self.deploy_batch_size_ = config[BATCH_SIZE_KEY]
		self.vector_merging_method_ = config[VECTOR_MERGING_METHOD_KEY]

	def caffe_tools(self):
		return self.caffe_tools_

	def caffe_data_dir(self):
		return self.caffe_data_dir_

	def portion_training(self):
		return self.portion_training_

	def deploy_mode(self):
		return self.deploy_mode_

	def deploy_model(self):
		return path.join(self.caffe_data_dir_, 'deploy.prototxt')

	def solver(self):
		return path.join(self.caffe_data_dir_, 'solver.prototxt')

	def mean(self):
		return np.load(path.join(self.caffe_data_dir_, 'mean.npy')).mean(1).mean(1)

	def finetuning_model(self):
		return self.caffe_finetuning_model_

	def caffe_model(self):
		return self.caffe_model_

	def deploy_batch_size(self):
		return self.deploy_batch_size_

	def vector_merging_method(self):
		return self.vector_merging_method_


