import os
import sys

def write_paths():
	try:
		start_path = sys.argv[1]
	except:
		start_path = "."
	path_list = open('paths', 'w')
	format = "_clean.jpg"
	for root, dirs, files in os.walk(start_path):
	    for name in files:
    		if name.endswith(format):
       			path_list.write(os.path.join(root, name) + "\n")
	    for name in dirs:
    		if name.endswith(format):
        		path_list.write(os.path.join(root, name) + "\n")
write_paths()
