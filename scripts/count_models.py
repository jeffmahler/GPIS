import os
import shutil
import sys

import IPython

#FORMATS = ['_clean.obj']
#FORMATS = ['.off']

if __name__ == '__main__':
    argc = len(sys.argv)
    root_dir = sys.argv[1]
    format = sys.argv[2]

    file_match = format
    if argc > 3:
        file_match = sys.argv[3]

    model_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(format):
                if f.find(file_match) != -1:
                    model_count = model_count + 1
                    
    print 'Num files in %s: %d' %(root_dir, model_count)
