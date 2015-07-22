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
        if len(files) == 2 and len(dirs) == 1:
            dir = dirs[0]
            common_prefix = os.path.commonprefix(files)
            postfixes = {f[len(common_prefix):] for f in files}
            if postfixes == {'_brute.pkl', '.json'}:
                # in directory with relevant files
                grasp_file = os.path.join(root, common_prefix + '.json')
                ua_result = os.path.join(root, common_prefix + '_brute.pkl')
                feature_dir = os.path.join(root, dir)
                print 'sudo cp -r %s %s %s %s' %(grasp_file, ua_result, feature_dir, target_dir)
