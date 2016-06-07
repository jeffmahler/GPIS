"""
Script to generate a new database of mesh and SDF models for grasping research.
Cleans the meshes, rescales, converts to obj, converts to sdf, extracts features, etc
TODO: Use stable poses and extract mesh categories

Author: Jeff Mahler
"""
import argparse
import IPython
import logging
import matplotlib.pyplot as plt
import os
import shutil
import sys

import category_map as catmap
import database as db
import experiment_config as ec
from mesh_file import MeshFile
from mesh_cleaner import MeshCleaner
import mesh_processor as mp
import obj_file
import stable_poses

# Global variables for conversion
class DatasetConfig:
    """ Struct to store relevant info about conversion for a particular dataset """
    def __init__(self, name, extension, name_filter = '', fix_names=False, scale=1.0, synthetic=True, cat_db=None):
        self.name = name               # name of dataset
        self.extension = extension     # default model extension for dataset
        self.name_filter = name_filter # template to search for a substring of the name
        self.fix_names = fix_names     # whether or not to rename the mesh
        self.scale = scale             # how much to scale the mesh to get to meters (because some objects are in mm instead of m)
        self.synthetic = synthetic     # whether or not the meshes are synthetic
        self.cat_db_ = cat_db          # dictionary of object categories
        self.count = 0                 # counter to determine how many models per dataset

    def set_count(self, count):
        """ Sets a new model count"""
        self.count = count

    def category(self, key):
        """ Returns the category for a given key """
        if self.cat_db_:
            return self.cat_db_.category(key)
        return ''

# Global array of all datasets and params
class DatasetConfigFactory:
    @staticmethod
    def available_datasets():
        return ['amazon_picking_challenge', 'aselab', 'autodesk', 'BigBIRD', 'Cat50_ModelDatabase',
                'dexnet_physical_experiments',
                'google', 'inventor_small', 'KIT', 'MeshSegBenchmark', 'ModelNet40', 'NTU3D',
                'PrincetonShapeBenchmark', 'SHREC14LSGTB', 'siemens', 'segments_small',
                'surgical', 'YCB']

    @staticmethod
    def config(name):
        if name not in DatasetConfigFactory.available_datasets():
            return None

        if name == 'amazon_picking_challenge':
            return DatasetConfig(name=name, extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=catmap.BerkeleyCategoryMap())
        elif name == 'aselab':
            return DatasetConfig(name=name, extension='.obj', scale=1.0, synthetic=False)
        elif name == 'autodesk':
            return DatasetConfig(name=name, extension='.off', synthetic=True)
        elif name == 'BigBIRD':
            return DatasetConfig(name=name, extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=catmap.BerkeleyCategoryMap())
        elif name == 'Cat50_ModelDatabase':
            return DatasetConfig(name=name, extension='.obj', cat_db=catmap.Cat50CategoryMap('/mnt/terastation/shape_data/Cat50_ModelDatabase'))
        elif name == 'dexnet_physical_experiments':
            return DatasetConfig(name=name, extension='.obj', synthetic=False)
        elif name == 'google':
            return DatasetConfig(name=name, extension='.obj', synthetic=True)
        elif name == 'inventor_small':
            return DatasetConfig(name=name, extension='.wrl', synthetic=True)
        elif name == 'KIT':
            return DatasetConfig(name=name, extension='.obj', name_filter='800_tex', scale=1e-3, synthetic=False)
        elif name == 'MeshSegBenchmark':
            return DatasetConfig(name=name, extension='.off', synthetic=True)
        elif name == 'ModelNet40':
            return DatasetConfig(name=name, extension='.off', cat_db=catmap.ModelNet40CategoryMap('/mnt/terastation/shape_data/MASTER_DB_v2/ModelNet40/index.db'))
        elif name == 'NTU3D':
            return DatasetConfig(name=name, extension='.obj')
        elif name == 'PrincetonShapeBenchmark':
            return DatasetConfig(name=name, extension='.off')
        elif name == 'SHREC14LSGTB':
            return DatasetConfig(name=name, extension='.off', cat_db=catmap.SHRECCategoryMap('/mnt/terastation/shape_data/SHREC14LSGTB/SHREC14LSGTB.cla'))
        elif name == 'siemens':
            return DatasetConfig(name=name, extension='.stl', synthetic=True)
        elif name == 'segments_small':
            return DatasetConfig(name=name, extension='.off', synthetic=True)
        elif name == 'surgical':
            return DatasetConfig(name=name, extension='.obj')
        elif name == 'YCB':
            return DatasetConfig(name=name, extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=catmap.BerkeleyCategoryMap())
        else:
            return None

# read in params
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/generate_database.yaml')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # filesystem params
    dataset = config['gen_dataset']
    shape_db_root_folder = config['shape_data_dir']
    dest_root_folder = config['database_dir']

    # get indices of dataset configurations
    dataset_start = 0
    if dataset != 'all':
        dataset_configs = [DatasetConfigFactory.config(dataset)]
    else:
        dataset_configs = [DatasetConfigFactory.config(name) for name in DatasetConfigFactory.available_datasets()]
    if dataset_configs[0] is None:
        raise Exception('Invalid dataset!')

    # create dest folder if doesn't exist
    if not os.path.exists(dest_root_folder):
        os.mkdir(dest_root_folder)

    # create list to store all exceptions
    exceptions = []

    # open up the database
    config['datasets'] = {}
    #[config['datasets'].update({c.name: []}) for c in dataset_configs]
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_WRITE_ACCESS)

    # loop through datasets and cleanup
    for dataset in dataset_configs:
        # setup next dataset
        all_filenames = []
        data_folder = os.path.join(shape_db_root_folder, dataset.name)
        data_list = os.listdir(data_folder)
        logging.info('Processing dataset %s' %(dataset.name))
        logging.info('Dataset scale=%f' %(dataset.scale))

        # create tmp dir for storing output of converter binaries
        target_dir = os.path.join(dest_root_folder, dataset.name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            os.system('chmod a+rwx %s' %(target_dir) )

        # create dest
        database.create_dataset(dataset.name)
        dataset_handle = database.dataset(dataset.name)

        # walk through the shape directory structure
        for root, sub_folders, files in os.walk(data_folder):
            # for each file in the subdir, look for valid models
            for f in files:
                # create file name
                filename = os.path.join(root, f)
                file_root, file_ext = os.path.splitext(f)
                fullpath_file_root = os.path.join(root, file_root)

                # grab the models with a given format and filtered by the given filter (fresh clean for each file)
                if file_ext == dataset.extension and filename.find(dataset.name_filter) > -1 and filename.find('clean') == -1\
                        and filename.find('dec') == -1 and filename.find('proc') == -1:
                    try:
                        mesh_processor = mp.MeshProcessor(filename)

                        config['obj_scale'] = dataset.scale
                        config['obj_rescaling_type'] = mp.MeshProcessor.RescalingTypeRelative
                        if dataset.synthetic:
                            config['obj_scale'] = config['gripper_size']
                            config['obj_rescaling_type'] = mp.MeshProcessor.RescalingTypeDiag

                        mesh_processor.generate_graspable(config)

                        # extract category
                        if dataset.fix_names:
                            category = dataset.category(root)
                            key = category
                        else:
                            category = dataset.category(file_root)
                            key = file_root

                        # mesh mass
                        mass = mesh_processor.mesh.mass
                        if mass < config['mass_thresh']:
                            mass = config['default_mass']

                        # write to database
                        logging.info('Creating graspable')
                        dataset_handle.create_graspable(key, mesh_processor.mesh, mesh_processor.sdf,
                                                        mesh_processor.shot_features,
                                                        mesh_processor.stable_poses,
                                                        category=category, mass=mass)
                    except Exception as e:
                        exceptions.append('Dataset: %s,  Model: %s, Exception: %s' % (dataset.name, filename, str(e))) 
                
    # print all exceptions
    exceptions_filename = os.path.join(dest_root_folder, 'exceptions.txt')
    out_exceptions = open(exceptions_filename, 'w')
    for exception in exceptions:
        out_exceptions.write('%s\n' %(exception))
    out_exceptions.close()

    # close the database
    database.close()
