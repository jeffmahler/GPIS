"""
Reads a file where lines are "ORIG_DATASET KEY [CATEGORY]" and creates a
directory of symlinks.

$ sudo python make_dataset.py train

Author: Brian Hou
"""

import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--terastation', action='store_true')
args = parser.parse_args()

if not args.terastation:
    database_path = '/home/brian/data'
    duplicate = os.symlink
else:
    database_path = '/mnt/terastation/shape_data/MASTER_DB_v1'
    duplicate = shutil.copyfile # can't make symlinks on terastation

index_db_lines = []
new_dataset_path = os.path.join(database_path, args.dataset)
new_feature_path = os.path.join(new_dataset_path, 'features')

with open(args.dataset) as f:
    os.mkdir(new_dataset_path)
    # os.mkdir(new_feature_path)
    for line in f:
        tokens = line.split()
        orig_dataset = tokens[0]
        key = tokens[1]
        category = '' if len(tokens) == 2 else tokens[2]

        old_dataset_path = os.path.join(database_path, orig_dataset)
        old_feature_path = os.path.join(old_dataset_path, 'features')

        # create symlinks
        # for ext in ('.obj', '.sdf', '.json'):
        for ext in ('.obj', '.sdf'):
            fname = key + ext
            src = os.path.join(old_dataset_path, fname)
            link_name = '%s_%s' %(orig_dataset, fname)
            dst = os.path.join(new_dataset_path, link_name)
            duplicate(src, dst)

        # symlink features
        # src = os.path.join(old_feature_path, key + '.json')
        # dst = os.path.join(new_feature_path, key + '.json')
        # duplicate(src, dst)

        # for src in glob.iglob(os.path.join(old_feature_path, key + '*')):
        #     dst = os.path.join(new_feature_path, os.path.basename(src))
        #     duplicate(src, dst)

        # add "origdataset_key category" to index_db_lines
        new_key = '%s_%s' %(orig_dataset, key)
        index_db_line = '%s %s\n' %(new_key, category) if category else new_key
        index_db_lines.append(index_db_line)

new_index = os.path.join(database_path, args.dataset, 'index.db')
with open(new_index, 'w') as f:
    f.writelines(index_db_lines)
