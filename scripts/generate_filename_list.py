import os
import sys

import IPython

FORMATS = ['.obj', '.skp', '.ply', '.off', '.3ds']

if __name__ == '__main__':
    root_dir = sys.argv[1]

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            for format in FORMATS:
                if f.endswith(format):
                    filename = os.path.join(root, f)
                    gs_cmd = 'gsutil cp %s gs://shape-database/%s' %(filename, filename[len(root_dir):])
                    os.system(gs_cmd)
                    print gs_cmd
