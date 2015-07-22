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

    for root, dirs, files in os.walk(args.result_dir):
        if len(files) == len(dirs) == 1:
            # in directory with relevant files
            file, dir = files[0], dirs[0]
            grasp_file = os.path.join(root, file)
            feature_dir = os.path.join(root, dir)
            print 'sudo cp -r %s %s %s' %(grasp_file, feature_dir, target_dir)
