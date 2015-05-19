import logging
import numpy as np
import os
import sys
import time

import experiment_config as ec
import graspable_object as go
import obj_file
import sdf_file

import IPython

INDEX_FILE = 'index.db'

class Database:
    def __init__(self, config):
        self._parse_config(config)
        self._create_datasets(config)
        
    def _parse_config(self, config):
        self.database_root_dir_ = config['database_dir']
        self.dataset_names_ = config['datasets']

    def _create_datasets(self, config):
        self.datasets_ = []
        for dataset_name in self.dataset_names_:
            self.datasets_.append(Dataset(dataset_name, config))

    @property
    def dataset_names(self):
        return self.dataset_names_

    @property
    def datasets(self):
        return self.datasets_

    def dataset(self, dataset_name=None):    
        if dataset_name is None:
            return self.datasets_.items()[0][0] # return first element
        return self.datasets_[dataset_name]

class Dataset:
    def __init__(self, dataset_name, config):
        self._parse_config(config)

        self.dataset_name_ = dataset_name
        self.dataset_root_dir_ = os.path.join(self.database_root_dir_, self.dataset_name_)
        self.iter_count_ = 0
        self.data_keys_ = []
        self.data_categories_ = []

        # read in filenames
        self._read_data_keys()

    def _parse_config(self, config):
        self.database_root_dir_ = config['database_dir']

    def _read_data_keys(self):
        """ Read in all the data keys from the index """
        index_filename = os.path.join(self.dataset_root_dir_, INDEX_FILE)
        if not os.path.exists(index_filename):
            raise IOError('Index file does not exist! Invalid dataset')

        self.data_keys_ = []
        self.data_categories_ = []
        index_file = open(index_filename, 'r')
        for line in index_file.readlines():
            tokens = line.split()
            self.data_keys_.append(tokens[0])

            if len(tokens) > 1:
                self.data_categories_.append(tokens[1])
            else:
                self.data_categories_.append('')

    @property
    def name(self):
        return self.dataset_name_

    @property
    def data_keys(self):
        return self.data_keys_

    @property
    def dataset_root_dir(self):
        return self.dataset_root_dir_

    @staticmethod
    def sdf_filename(file_root):
        return file_root + '.sdf'

    @staticmethod
    def obj_filename(file_root):
        return file_root + '.obj'

    def read_datum(self, key):
        """ Read in the datapoint corresponding to given key"""
        # get file roots
        file_root = os.path.join(self.dataset_root_dir_, key)
        sdf_filename = Dataset.sdf_filename(file_root)
        obj_filename = Dataset.obj_filename(file_root)

        # read in data
        sf = sdf_file.SdfFile(sdf_filename)
        sdf = sf.read()

        of = obj_file.ObjFile(obj_filename)
        mesh = of.read()

        return go.GraspableObject3D(sdf, mesh=mesh, key=key)

    def __iter__(self):
        """ Generate iterator """
        self.iter_count_ = 0 # NOT THREAD SAFE!
        return self

    def next(self):
        """ Read the next object file in the list """
        if self.iter_count_ >= len(self.data_keys_):
            raise StopIteration
        else:
            logging.info('Returning datum %s' %(self.data_keys_[self.iter_count_]))
            obj = self.read_datum(self.data_keys_[self.iter_count_])
            self.iter_count_ = self.iter_count_ + 1
            return obj

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = 'cfg/basic_labelling.yaml'
    config = ec.ExperimentConfig(config_filename)

    db = Database(config)
    keys = []
    logging.info('Reading datset %s' %(db.datasets[0].name))
    for obj in db.datasets[0]:
        keys.append(obj.key)

    assert(len(keys) == 26)
