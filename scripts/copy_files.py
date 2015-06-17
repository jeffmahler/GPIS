import os
import shutil
import sys

import IPython

FORMATS = ['.obj', '.sdf']

if __name__ == '__main__':
    argc = len(sys.argv)
    root_dir = sys.argv[1]
    target_dir = sys.argv[2]
    file_match = sys.argv[3]

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            for format in FORMATS:
                if f.endswith(format):
                    cur_filename = os.path.join(root, f)
                    if cur_filename.find(file_match) == -1:
                        dirs, file_root = os.path.split(cur_filename)
                        head, obj = os.path.split(dirs)
#                        head, obj = os.path.split(head)

                        target_subdir = os.path.join(target_dir, obj)
                        target_filename = os.path.join(target_subdir, f)
                        
                        if not os.path.exists(target_subdir):
                            os.mkdir(target_subdir)
                        shutil.copy(cur_filename, target_filename)
                        print target_filename
