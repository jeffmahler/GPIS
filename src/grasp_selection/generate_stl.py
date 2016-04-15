"""
Script to generate .STL files for a dataset for 3D printing
Author: Jeff Mahler
"""
import logging
import os
import sys

import database as db
import experiment_config as ec

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # open config and read params
    config = ec.ExperimentConfig(config_filename)
    config['database_cache_dir'] = output_dir
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    scale = 1.0
    if config['scale'] is not None:
        scale = config['scale']

    # open database and dataset
    database = db.Hdf5Database(database_filename, config)
    ds = database.dataset(dataset_name)

    # convert each object
    for obj in ds:
        logging.info('Converting object %s to STL' %(obj.key))
        stl_filename = ds.stl_mesh_filename(obj.key, scale=scale)

