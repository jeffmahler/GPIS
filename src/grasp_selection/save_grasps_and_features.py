import argparse
import logging
import os

import IPython
import experiment_config as ec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('result_dir')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    config = ec.ExperimentConfig(args.config)
    target_dir = os.path.join(config['database_dir'], config['dataset'])
    json = os.path.join(args.result_dir, '*/cm_out/test_data/*.json')
    features = os.path.join(args.result_dir, '*/cm_out/test_data/features')
    logging.info('Run this command yourself:')
    print 'sudo cp -r %s %s %s' %(json, features, target_dir)
