"""
YAML Configuration Parser - basically reads everything into a dictionary

Author : Jeff Mahler
"""
import argparse
import logging
import math
import numpy as np
import os
import scipy
import shutil
import sys
import time

import yaml, re
from collections import OrderedDict
import os.path as osp

import IPython

# template replacement for prototxts
# sed s/<TEMPLATE>/<REPLACEMENT>/ <OLD_FILE> ><NEW_FILE>
TEMPLATE_REPLACE_COMMAND = 'sed %s %s >%s'
TEMPLATE_SINGLE_REPLACE = 's/%s/%s/'

# template leveldb conversion
TEMPLATE_LEVELDB_COMMAND = '%s %s %s %s %s %s'

# default access keys
SOURCE_KEY = 'source'
OUTPUT_KEY = 'output'
TEMPLATE_KEY = 'templates'

LEVELDB_KEY = 'leveldbs'

SCRIPT_KEY = 'script'
DATA_KEY = 'data'
DB_KEY = 'db'
NUM_KEY = 'num'
SAMPLE_KEY = 'sample'

# hardcoded params that are functions of other params
DECODER_DEPLOY_BATCH_SIZE_EXPR = '2*config[\'hidden_state_dim\']' # hanrcoded param
OBSERVATION_DIM_EXPR = 'config[\'width\']*config[\'height\']' # hanrcoded param

class ExperimentConfig(object):
    """
    Class to load a configuration file, parse config, fill templates, and create necessary I/O dirs / databases
    All configs may have a comon root_dir key for easy file access
    """
    def __init__(self, filename = None, use_templates=False, create_dbs=False):
        self.config = None # initialize empty config
        self.use_templates = use_templates
        self.filename = filename

        if filename is not None:
            self.load_config(filename)

    def load_config(self, filename):
        """
        Loads a yaml configuration file from the given filename
        """
        # read entire file for metadata
        fh = open(filename, 'r')
        self.file_contents = fh.read()

        # replace !include directives with content
        config_dir = os.path.split(filename)[0]
        include_re = re.compile('^!include\s+(.*)$', re.MULTILINE)
        def include_repl(matchobj):
            fname = os.path.join(config_dir, matchobj.group(1))
            with open(fname) as f:
                return f.read()
        while re.search(include_re, self.file_contents): # for recursive !include
            self.file_contents = re.sub(include_re, include_repl, self.file_contents)

        # read in dictionary
        self.config = self.__ordered_load(self.file_contents)

        # convert functions of other params to true expressions
        for k in self.config.keys():
            self.config[k] = ExperimentConfig.__convert_key(self.config[k])

        # load core configuration
        try:
            self.root_dir = self.config['root_dir']
        except KeyError:
            self.root_dir = '' # relative paths
        return self.config

    def __contains__(self, key):
        return key in self.config

    def __getitem__(self, key):
        """
        Overrides the key access operator []
        """
        try:
            retval = self.config[key]
        except KeyError:
            logging.warning('Key %s does not exist. Returning None.' %(key))
            return None
        return retval

    def __setitem__(self, key, val):
        self.config[key] = val

    @staticmethod
    def __convert_key(expression):
        """ Converts keys in YAML that reference other keys """
        if type(expression) is str and len(expression) > 2 and expression[1] == '!':
            expression = eval(expression[2:-1])
        return expression

    def __ordered_load(self, stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        """
        Load an ordered dictionary from a yaml file. Borrowed from John Schulman

        See:
        http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts/21048064#21048064"
        """
        class OrderedLoader(Loader):
            pass
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: object_pairs_hook(loader.construct_pairs(node)))
        return yaml.load(stream, OrderedLoader)

    def fill_templates(self):
        """
        Fills all template requests in the input configuration file
        """
        for k in self.templates.keys():
            try:
                source_filename = self.templates[k][SOURCE_KEY]
                out_filename    = self.templates[k][OUTPUT_KEY]
                template_pairs  = self.templates[k][TEMPLATE_KEY]
                self.__fill_template(source_filename, out_filename, template_pairs)
            except KeyError:
                logging.warning('Could not parse template block of configuration file.')

    def __fill_template(self, template_filename, out_filename, template_pairs):
        """
        Replaces all occurrences of template with value in template_filename, saves result to out_filename.
        This is intended to speed up prototxt configuration for caffe
        Uses sed from the command line
        """
        template_filename_path = os.path.join(self.root_dir, template_filename)
        out_filename_path = os.path.join(self.root_dir, out_filename)

        # check for existence of files, notify user
        if not os.path.exists(template_filename_path):
            logging.warning('Template file %s does not exist' %(template_filename_path))
            return False

        if os.path.exists(out_filename_path):
            logging.warning('Output file %s will be overwritten' %(out_filename_path))

        replacements = '\''
        for (t, r) in template_pairs.items():
            t = self.__convert_key(t)
            r = self.__convert_key(r)
            if type(t) is str:
                t = t.replace('/', '\/')
            if type(r) is str:
                r = r.replace('/', '\/')
            next_replace = TEMPLATE_SINGLE_REPLACE %(t, r)
            replacements += next_replace + '; '
        replacements += '\''

        replace_command = TEMPLATE_REPLACE_COMMAND %(replacements, template_filename_path, out_filename_path)
        logging.info('Command: %s' %(replace_command))
        os.system(replace_command)
        shutil.copymode(template_filename_path, out_filename_path) #copy over permission
        return True

    def create_leveldbs(self):
        """
        Creates all leveldbs specified in the configuration for Caffe.
        Currentl deprecated but could be reinstated if useful in future experiments
        """
        try:
            leveldbs = self.config[LEVELDB_KEY]
        except KeyError:
            logging.warning('Cannot create leveldbs - none specified in configuration')
            return False

        for k in leveldbs:
            try:
                script = leveldbs[k][SCRIPT_KEY]
                data   = leveldbs[k][DATA_KEY]
                db     = leveldbs[k][DB_KEY]
                num    = leveldbs[k][NUM_KEY]
                sample = leveldbs[k][SAMPLE_KEY]
                self.__create_leveldb(script, data, db, num, sample)
            except KeyError:
                logging.warning('Could not parse leveldb block of configuration file.')

    def __create_leveldb(self, script, data, database_name, num_to_convert, sample):
        """
        Creates leveldbs from configuration data. This is specific to caffe and should not be considered general.
        Assumes database name is a root for the three databases: -data, -tar, -state
        """
        # get the caffe root directory
        try:
            caffe_root = self.config['caffe_root']
        except KeyError:
            caffe_root = './'

        leveldb_os_call = os.path.join(caffe_root, script)
        data_path = os.path.join(self.root_dir, data)
        database_path = os.path.join(self.root_dir, database_name) + '-' # hyphen to meet naming convention

        # create specific paths
        data_database_path = database_path + 'data'
        tar_database_path = database_path + 'tar'
        state_database_path = database_path + 'state'

        # remove old leveldbs
        if os.path.exists(data_database_path):
            logging.warning('Database %s will be overwritten' %(data_database_path))
            shutil.rmtree(data_database_path)

        if os.path.exists(tar_database_path):
            logging.warning('Database %s will be overwritten' %(tar_database_path))
            shutil.rmtree(tar_database_path)

        if os.path.exists(state_database_path):
            logging.warning('Database %s will be overwritten' %(state_database_path))
            shutil.rmtree(state_database_path)

        caffe_root_patched = caffe_root
        if caffe_root_patched[-1] != '/':
            caffe_root_patched += '/'

        # add 1 to num_to_convert because of the way the file is structured
        # TODO: fix and remove +1
        data_to_leveldb_command = TEMPLATE_LEVELDB_COMMAND %(leveldb_os_call, caffe_root_patched, data_path, database_path, (num_to_convert+1), sample)
        logging.info('Command: %s' %(data_to_leveldb_command))
        os.system(data_to_leveldb_command)

def test_load():
    filename = 'data/test/test_config.yaml'
    ec = ExperimentConfig(filename)

    logging.debug('Data directory: %s' %(ec['data_dir']))
    logging.debug('Num pose samples: %d' %(ec['num_pose_samples']))
    logging.debug('Translation sigma: %f' %(ec['translation_sigma']))
    logging.debug('Rotation sigma: %f' %(ec['rotation_sigma']))

    assert(ec['data_dir'] == '/mnt/terastation/shape_data')
    assert(ec['num_pose_samples'] == 100)
    assert(ec['translation_sigma'] == 0.1)
    assert(ec['rotation_sigma'] == 0.01)
    logging.debug('TEST PASSED!')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_load()
