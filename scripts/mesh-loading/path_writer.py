import os
import sys

def write_paths():
	try:
		start_path = sys.argv[1]
	except:
		start_path = "."
	print(start_dir)
	path_list = open('paths', 'w')
	formats = [".obj", ".skp", ".3DS", ".off", ".h5"]
	for root, dirs, files in os.walk(start_path):
	    for name in files:
	    	for format in formats:
	    		if name.endswith(format):
	       			path_list.write(os.path.join(root, name) + "\n")
	    for name in dirs:
	    	for format in formats:
	    		if name.endswith(format):
	        		path_list.write(os.path.join(root, name) + "\n")
write_paths()
