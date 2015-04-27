import os
import shutil
import sys

import IPython

FORMATS = ['.obj']

if __name__ == '__main__':
    argc = len(sys.argv)
    root_dir = sys.argv[1]
    
    dim = 25
    if argc > 2:
        dim = int(sys.argv[2])

    padding = 5
    if argc > 3:
        padding = int(sys.argv[3])

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            for format in FORMATS:
                if f.endswith(format):
                    filename = os.path.join(root, f)
                    fileroot, file_ext = os.path.splitext(filename)
                    sdf_filename = fileroot + '.sdf'
                    sdf_dim_filename = fileroot + '_%d'%(dim) + '.sdf'
                    sdfgen_cmd = '/home/jmahler/Libraries/SDFGen/bin/SDFGen %s %d %d' %(filename, dim, padding)
                    os.system(sdfgen_cmd)
                    shutil.move(sdf_filename, sdf_dim_filename)
                    print sdfgen_cmd
