"""
Reads a file where lines are "ORIG_DATASET KEY [CATEGORY]" and creates a
directory of symlinks.

$ python make_dataset.py train \
    --database-path /mnt/terastation/shape_data/MASTER_DB_v1/

Author: Brian Hou
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--database-path', default='/home/brian/data')
args = parser.parse_args()

index_db_lines = []
with open(args.dataset) as f:
    os.mkdir(args.dataset)
    for line in f:
        tokens = line.split()
        orig_dataset = tokens[0]
        key = tokens[1]
        category = '' if len(tokens) == 2 else tokens[2]

        # create symlink
        src = os.path.join(args.database_path, orig_dataset, key)
        link_name = '%s_%s' %(orig_dataset, key)
        dst = os.path.join(args.database_path, args.dataset, link_name)
        os.symlink(src, dst)

        # add "origdataset_key category" to index_db_lines
        index_db_line = '%s %s' %(link_name, category) if category else link_name
        index_db_lines.append(index_db_line)

new_index = os.path.join(args.database_path, args.dataset, 'index.db')
with open(new_index, 'w') as f:
    f.writelines(index_db_lines)
