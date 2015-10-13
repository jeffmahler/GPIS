import database as db
import experiment_config as ec
import logging
import os
import sys

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/correlated.yaml')
    parser.add_argument('output_dest', default='out/')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # make output directory
    dest = config['database_dir']
    filename = os.path.join(dest, 'test.txt')
    f = open(filename, 'w')
    f.write('wrote to db\n')
    f.close()
