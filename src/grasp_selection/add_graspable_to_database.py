"""
Script to add a new graspable to dataset from a single mesh file
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


# Global array of all datasets and params
ILLEGAL_DATASETS = ['BigBIRD', 'Cat50_ModelDatabase',
                    'KIT', 'MeshSegBenchmark', 'ModelNet40', 'NTU3D',
                    'PrincetonShapeBenchmark', 'SHREC14LSGTB', 'YCB']

# read in params
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file')
    parser.add_argument('config')
    parser.add_argument('--synthetic', default=1)    
    parser.add_argument('--category', default='unknown')    
    parser.add_argument('--scale', default=1.0)    
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # filesystem params
    mesh_filename = args.mesh_file
    dataset_name = config['gen_dataset']
    shape_db_root_folder = config['shape_data_dir']
    dest_root_folder = config['database_dir']
    synthetic = int(args.synthetic)
    category = args.category
    scale = float(args.scale)

    if dataset_name in ILLEGAL_DATASETS:
        raise ValueError('Cannot add a graspable to dataset %s' %(dataset_name))

    # open database
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_WRITE_ACCESS)
    dataset = database.dataset(dataset_name)

    # create tmp dir for storing output of converter binaries
    target_dir = os.path.join(dest_root_folder, dataset.name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        os.system('chmod a+rwx %s' %(target_dir) )

    # parse filename
    file_path, file_name = os.path.split(mesh_filename)
    file_root, file_ext = os.path.splitext(file_name)
    key = file_root

    # grab the models with a given format and filtered by the given filter (fresh clean for each file)
    mesh_processor = mp.MeshProcessor(mesh_filename)

    # generate a graspable
    config['scale'] = scale
    config['obj_rescaling_type'] = mp.MeshProcessor.RescalingTypeAbsolute
    if synthetic == 1:
        config['obj_scale'] = config['gripper_size']
        config['obj_rescaling_type'] = mp.MeshProcessor.RescalingTypeDiag
    mesh_processor.generate_graspable(config)
    
    # mesh mass
    mass = mesh_processor.mesh.mass
    if mass < config['mass_thresh']:
        mass = config['default_mass']
            
    # write to database
    logging.info('Creating graspable')
    dataset.create_graspable(key, mesh_processor.mesh, mesh_processor.sdf,
                             mesh_processor.shot_features,
                             mesh_processor.stable_poses,
                             category=category, mass=mass)

    # close the database
    database.close()
