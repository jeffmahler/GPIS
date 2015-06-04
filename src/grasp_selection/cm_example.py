"""
Toy example to find the center of mass of objects.

$ python cm_example.py [config] [output_dest]

Author: Brian Hou
"""

import argparse
import os
import logging
import IPython

import experiment_config as ec
import database as db

def save_result(obj, result, output_dest):
    fname = os.path.join(output_dest, obj.key + '.out')
    with open(fname, 'w') as f:
        f.write(str(result))

def center_of_mass(graspable):
    return list(graspable.center_of_mass_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/example.yaml')
    parser.add_argument('output_dest', default='out/')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # make output directory
    try:
        os.makedirs(args.output_dest)
    except os.error:
        pass

    # loop through objects, finding CM for each
    database = db.Database(config)
    for dataset in database.datasets:
        logging.info('Labelling dataset {}'.format(dataset.name))
        for obj in dataset:
            logging.info('Labelling object {}'.format(obj.key))

            result = center_of_mass(obj)
            save_result(obj, result, args.output_dest)
