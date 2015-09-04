import os
import sys

rm_dir = sys.argv[1]
files = os.listdir(rm_dir)
for f in files:
    filename = os.path.join(rm_dir, f)
    if os.path.islink(filename):
        print 'Removing', filename
        os.remove(filename)
