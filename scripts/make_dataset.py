"""
Reads a file where lines are "ORIG_DATASET KEY [CATEGORY]" and creates a
directory of symlinks.

$ sudo python make_dataset.py train

Author: Brian Hou
"""

import argparse
import os

DATABASE_PATH = '/home/brian/data'

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

index_db_lines = []
with open(args.dataset) as f:
    os.mkdir(os.path.join(DATABASE_PATH, args.dataset))
    for line in f:
        tokens = line.split()
        orig_dataset = tokens[0]
        key = tokens[1]
        category = '' if len(tokens) == 2 else tokens[2]

        # create symlink
        src = os.path.join(DATABASE_PATH, orig_dataset, key)
        link_name = '%s_%s' %(orig_dataset, key)
        dst = os.path.join(DATABASE_PATH, args.dataset, link_name)
        os.symlink(src, dst)

        # add "origdataset_key category" to index_db_lines
        index_db_line = '%s %s\n' %(link_name, category) if category else link_name
        index_db_lines.append(index_db_line)

new_index = os.path.join(DATABASE_PATH, args.dataset, 'index.db')
with open(new_index, 'w') as f:
    f.writelines(index_db_lines)
