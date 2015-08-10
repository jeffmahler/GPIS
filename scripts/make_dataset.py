"""
Reads a file where lines are "ORIG_DATASET KEY [CATEGORY]" and creates a
directory of symlinks.

$ sudo python make_dataset.py train

Author: Brian Hou
"""

import argparse
import glob
import os

DATABASE_PATH = '/home/brian/data'

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

index_db_lines = []
new_dataset_path = os.path.join(DATABASE_PATH, args.dataset)
new_feature_path = os.path.join(new_dataset_path, 'features')

with open(args.dataset) as f:
    os.mkdir(new_dataset_path)
    # os.mkdir(new_feature_path)
    for line in f:
        tokens = line.split()
        orig_dataset = tokens[0]
        key = tokens[1]
        category = '' if len(tokens) == 2 else tokens[2]

        old_dataset_path = os.path.join(DATABASE_PATH, orig_dataset)
        old_feature_path = os.path.join(old_dataset_path, 'features')

        # create symlinks
        # for ext in ('.obj', '.sdf', '.json'):
        for ext in ('.obj', '.sdf'):
            fname = key + ext
            src = os.path.join(old_dataset_path, fname)
            link_name = '%s_%s' %(orig_dataset, fname)
            dst = os.path.join(new_dataset_path, link_name)
            os.symlink(src, dst)

        # symlink features
        # src = os.path.join(old_feature_path, key + '.json')
        # dst = os.path.join(new_feature_path, key + '.json')
        # os.symlink(src, dst)

        # for src in glob.iglob(os.path.join(old_feature_path, key + '*')):
        #     dst = os.path.join(new_feature_path, os.path.basename(src))
        #     os.symlink(src, dst)

        # add "origdataset_key category" to index_db_lines
        new_key = '%s_%s' %(orig_dataset, key)
        index_db_line = '%s %s\n' %(new_key, category) if category else new_key
        index_db_lines.append(index_db_line)

new_index = os.path.join(DATABASE_PATH, args.dataset, 'index.db')
with open(new_index, 'w') as f:
    f.writelines(index_db_lines)
