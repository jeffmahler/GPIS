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
    chunk = db.Chunk(config)

    # make output directory
    try:
        dest = os.path.join(args.output_dest, chunk.name)
        os.makedirs(dest)
    except os.error:
        pass

    # loop through objects, finding CM for each
    for obj in chunk:
        logging.info('Labelling object {}'.format(obj.key))
        result = center_of_mass(obj)
        save_result(obj, result, dest)
